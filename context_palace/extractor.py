"""
IncrementalDeltaExtractor for parsing conversational turns into ConceptNodes and RelationEdges,
performing vector/canonical deduplication, watermarking turns, and resolving conflicts via ConsensusEngine.
"""

from __future__ import annotations

import inspect
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from context_palace.consensus import ConsensusEngine
from context_palace.models import (
    AgentSession,
    BeliefRevision,
    ConceptNode,
    ConflictResolutionEvent,
    RelationEdge,
    compute_text_embedding,
    cosine_similarity,
    estimate_tokens,
)

DISTINCT_ATTRIBUTE_TOKENS: Set[str] = {
    "host",
    "port",
    "url",
    "uri",
    "user",
    "username",
    "password",
    "secret",
    "key",
    "token",
    "timeout",
    "ttl",
    "limit",
    "size",
    "path",
    "dir",
    "mode",
    "version",
    "provider",
    "model",
    "driver",
    "backend",
    "frontend",
    "protocol",
    "region",
}

STOPWORD_PREFIXES: Tuple[str, ...] = (
    "the ",
    "a ",
    "an ",
    "our ",
    "current ",
    "default ",
    "active ",
    "primary ",
    "main ",
    "system ",
)

PROVENANCE_DEFAULT_CONFIDENCE: Dict[str, float] = {
    "operator_crud": 0.99,
    "user_explicit": 0.95,
    "verified_test": 0.92,
    "agent_turn": 0.85,
    "model_inference": 0.70,
}


class IncrementalDeltaExtractor:
    """
    Extracts structured ConceptNodes and RelationEdges from unstructured agent turns,
    deduplicates concepts via canonical & vector similarity, enforces incremental
    turn watermarking on AgentSession, and resolves conflicting beliefs via ConsensusEngine.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.78,
        consensus_engine: Optional[ConsensusEngine] = None,
    ) -> None:
        self.similarity_threshold = float(similarity_threshold)
        self.consensus_engine = consensus_engine or ConsensusEngine()
        # Tracks supporting agent IDs per (node_id, normalized_value) for multi-agent vote weighting
        self._value_voters: Dict[Tuple[str, str], Set[str]] = {}
        # Fallback provenance tracker for nodes when lake.get_belief_history is empty
        self._node_provenance: Dict[str, str] = {}

    @staticmethod
    def _normalize_name(raw: str) -> str:
        """Normalize a concept key into a canonical lowercase snake_case identifier."""
        cleaned = re.sub(r"[*`#~>\[\]()\"']", "", raw or "").strip().lower()
        cleaned = re.sub(r"^[-•+\d.)\s]+", "", cleaned).strip()
        cleaned = re.sub(r"[_/\-]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        changed = True
        while changed:
            changed = False
            for prefix in STOPWORD_PREFIXES:
                if cleaned.startswith(prefix) and len(cleaned) > len(prefix) + 2:
                    cleaned = cleaned[len(prefix) :].strip()
                    changed = True
        cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
        # Normalize common synonyms for canonical concept matching
        synonym_map = {
            "db": "database",
            "database_engine": "database",
            "db_engine": "database",
            "database_system": "database",
            "llm_model": "model",
            "ai_model": "model",
            "server_port": "port",
            "http_port": "port",
            "service_port": "port",
        }
        return synonym_map.get(cleaned, cleaned)

    @staticmethod
    def _normalize_value(val: str) -> str:
        cleaned = re.sub(r"[*_`\"']", "", val or "").strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.rstrip(".,;").strip()

    @classmethod
    def _classify_category(cls, canonical_name: str, active_value: str, raw_clause: str = "") -> str:
        combined = f"{canonical_name} {active_value} {raw_clause}".lower()
        name_lower = canonical_name.lower()

        if any(
            kw in name_lower or kw in combined
            for kw in ("constraint", "must_not", "never", "forbid", "invariant", "sla", "strict", "disallow")
        ):
            return "constraint"
        if any(
            kw in name_lower
            for kw in (
                "port",
                "host",
                "url",
                "uri",
                "env",
                "flag",
                "token",
                "ttl",
                "timeout",
                "limit",
                "size",
                "path",
                "dir",
                "mode",
                "version",
                "model",
                "retry",
                "config",
                "setting",
            )
        ):
            return "config"
        if any(kw in name_lower for kw in ("decision", "adr", "strategy", "policy", "choice", "selected")):
            return "decision"
        if any(
            kw in combined
            for kw in (
                "database",
                "postgres",
                "sqlite",
                "redis",
                "kafka",
                "rabbitmq",
                "service",
                "gateway",
                "cache",
                "queue",
                "broker",
                "engine",
                "storage",
                "architecture",
                "stack",
                "framework",
                "backend",
                "frontend",
                "api",
                "cluster",
                "mesh",
                "lake",
                "depends_on",
                "connects_to",
                "uses",
            )
        ):
            return "architecture"
        return "entity"

    def extract_from_text(
        self,
        text: str,
        source_agent_id: str = "agent",
        provenance_type: str = "agent_turn",
        turn_id: int = 0,
        timestamp: Optional[float] = None,
    ) -> Tuple[List[ConceptNode], List[RelationEdge]]:
        """
        Deterministically parse unstructured conversational turns or markdown specs
        into ConceptNode items and RelationEdge relationships.
        """
        if not text or not text.strip():
            return [], []

        now = float(timestamp) if timestamp is not None else time.time()
        default_conf = PROVENANCE_DEFAULT_CONFIDENCE.get(provenance_type, 0.85)

        nodes_by_name: Dict[str, ConceptNode] = {}
        edges: List[RelationEdge] = []
        edge_keys: Set[Tuple[str, str, str]] = set()

        def _upsert_local_node(
            raw_key: str,
            raw_val: str,
            clause: str = "",
            override_category: Optional[str] = None,
        ) -> Optional[ConceptNode]:
            c_name = self._normalize_name(raw_key)
            c_val = self._normalize_value(raw_val)
            if not c_name or not c_val or len(c_name) < 2:
                return None
            category = override_category or self._classify_category(c_name, c_val, clause)
            emb = compute_text_embedding(f"{c_name} {c_val}")
            summary = clause.strip() if clause.strip() else f"{c_name}: {c_val}"
            if len(summary) > 160:
                summary = summary[:157] + "..."

            if c_name in nodes_by_name:
                existing = nodes_by_name[c_name]
                existing.active_value = c_val
                existing.category = category
                existing.embedding = emb
                existing.summary = summary
                existing.updated_at = now
                return existing

            node = ConceptNode(
                node_id=f"node_{uuid.uuid4().hex[:12]}",
                canonical_name=c_name,
                category=category,
                active_value=c_val,
                summary=summary,
                confidence=default_conf,
                pinned=(provenance_type == "operator_crud"),
                embedding=emb,
                source_agent_id=source_agent_id,
                created_at=now,
                updated_at=now,
                tombstoned=False,
            )
            nodes_by_name[c_name] = node
            return node

        def _add_local_edge(src_id: str, tgt_id: str, rel_type: str, weight: float = 1.0) -> None:
            if not src_id or not tgt_id or src_id == tgt_id:
                return
            key = (src_id, tgt_id, rel_type)
            if key in edge_keys:
                return
            edge_keys.add(key)
            edges.append(
                RelationEdge(
                    edge_id=f"edge_{uuid.uuid4().hex[:12]}",
                    source_id=src_id,
                    target_id=tgt_id,
                    relation_type=rel_type,
                    weight=weight,
                    source_agent_id=source_agent_id,
                    valid_from=now,
                )
            )

        # Split into lines and sentence clauses without breaking version numbers or IPs
        noise_prefixes = (
            "<div",
            "</div",
            "created at:",
            "completed at:",
            "command exited with code",
            "command output:",
            "showing lines ",
            "task logs are available at:",
            "the tool's output was truncated",
            "here is a preview",
            "enumerating objects:",
            "counting objects:",
            "compressing objects:",
            "writing objects:",
            "remote: resolving deltas:",
            "total ",
            "working...",
        )
        raw_lines: List[str] = []
        for ln in text.splitlines():
            s_ln = ln.strip()
            if not s_ln:
                continue
            low_ln = s_ln.lower()
            if low_ln.startswith(noise_prefixes):
                continue
            if re.match(r"^\d+:\s", s_ln):
                continue
            if "style=" in low_ln or "rgba(" in low_ln:
                continue
            raw_lines.append(s_ln)

        clauses: List[str] = []
        for line in raw_lines:
            parts = re.split(r"(?:;\s+|\.\s+(?=[A-Z*`]))", line)
            for part in parts:
                cleaned_part = part.strip().rstrip(".")
                if cleaned_part:
                    clauses.append(cleaned_part)

        kv_pattern = re.compile(
            r"^(?:[-*•+]\s*)?(?:\*\*|__|`)*([A-Za-z][A-Za-z0-9_/\-\s]{1,45}?)(?:\*\*|__|`)*\s*(?::|=)\s*(.+)$"
        )
        rel_pattern = re.compile(
            r"^(?:[-*•+]\s*)?(?P<src>[A-Za-z][A-Za-z0-9_\-\s]{1,35}?)\s+"
            r"(?P<verb>uses|depends on|connects to|configures|implements|routes to|is backed by|calls|requires)\s+"
            r"(?P<tgt>[A-Za-z0-9_.\-/: ]+)$",
            re.IGNORECASE,
        )
        copula_pattern = re.compile(
            r"^(?:[-*•+]\s*)?(?:we\s+use\s+|use\s+)?(?P<key>[A-Za-z][A-Za-z0-9_\-\s]{1,35}?)\s+"
            r"(?:is\s+set\s+to|is\s+configured\s+to|is\s+configured\s+as|must\s+be|should\s+be|will\s+be|is|are)\s+"
            r"(?P<val>[^,.;]+)$",
            re.IGNORECASE,
        )

        verb_to_relation = {
            "uses": "depends_on",
            "depends on": "depends_on",
            "requires": "depends_on",
            "is backed by": "depends_on",
            "connects to": "connects_to",
            "routes to": "connects_to",
            "calls": "connects_to",
            "configures": "configures",
            "implements": "implements",
        }

        for clause in clauses:
            matched_clause = False

            # 1. Explicit Key: Value or Key = Value pattern
            m_kv = kv_pattern.match(clause)
            if m_kv and not clause.lower().startswith(("http://", "https://")):
                k_raw, v_raw = m_kv.group(1).strip(), m_kv.group(2).strip()
                override_cat = "constraint" if k_raw.lower().startswith("constraint") else None
                if override_cat == "constraint" and k_raw.lower() == "constraint":
                    # Derive specific constraint name from value tokens
                    val_words = re.findall(r"[a-zA-Z0-9]+", v_raw.lower())[:4]
                    k_raw = "constraint_" + "_".join(val_words) if val_words else "constraint"
                node = _upsert_local_node(k_raw, v_raw, clause, override_category=override_cat)
                if node is not None:
                    matched_clause = True

            # 2. Relational architectural statement ("X uses Y", "X depends on Y", "X connects to Y")
            if not matched_clause:
                m_rel = rel_pattern.match(clause)
                if m_rel:
                    src_raw = m_rel.group("src").strip()
                    verb_raw = m_rel.group("verb").strip().lower()
                    tgt_raw = m_rel.group("tgt").strip()
                    src_node = _upsert_local_node(
                        src_raw,
                        f"{verb_raw} {tgt_raw}",
                        clause,
                        override_category="architecture",
                    )
                    tgt_node = _upsert_local_node(
                        tgt_raw,
                        tgt_raw,
                        clause,
                        override_category="entity",
                    )
                    if src_node and tgt_node:
                        rel_type = verb_to_relation.get(verb_raw, "depends_on")
                        _add_local_edge(src_node.node_id, tgt_node.node_id, rel_type)
                        matched_clause = True

            # 3. Declarative copula statement ("Database is PostgreSQL", "Port is 8080")
            if not matched_clause:
                m_cop = copula_pattern.match(clause)
                if m_cop:
                    k_raw = m_cop.group("key").strip()
                    v_raw = m_cop.group("val").strip()
                    node = _upsert_local_node(k_raw, v_raw, clause)
                    if node is not None:
                        matched_clause = True

            # 4. Fallback salient technical clause extraction if no explicit pattern matched in short turns
            if not matched_clause and not nodes_by_name and len(clauses) <= 4 and len(clause) >= 12:
                words = [
                    w.lower()
                    for w in re.findall(r"[A-Za-z][A-Za-z0-9_\-]*", clause)
                    if w.lower() not in {"the", "and", "for", "with", "that", "this", "from", "into", "will", "must"}
                ]
                if len(words) >= 2:
                    fallback_key = "_".join(words[:3])
                    _upsert_local_node(fallback_key, clause, clause)

        nodes = list(nodes_by_name.values())

        # Co-occurrence edges between concepts extracted in the same turn
        if len(nodes) >= 2:
            primary = nodes[0]
            for secondary in nodes[1:]:
                rel_type = "configures" if secondary.category == "config" else "depends_on"
                _add_local_edge(primary.node_id, secondary.node_id, rel_type, weight=0.85)

        return nodes, edges

    def find_matching_concept(
        self,
        candidate_name: str,
        candidate_value: str,
        existing_nodes: List[ConceptNode],
    ) -> Optional[ConceptNode]:
        """
        Find an existing ConceptNode matching candidate_name:
          1. Exact normalized canonical_name match.
          2. Token/substring overlap on canonical_name or cosine similarity >= similarity_threshold
             between compute_text_embedding(candidate_name) and compute_text_embedding(node.canonical_name).
        """
        if not existing_nodes:
            return None

        cand_norm = self._normalize_name(candidate_name)
        if not cand_norm:
            return None

        active_nodes = [n for n in existing_nodes if not getattr(n, "tombstoned", False)]

        # 1. Exact normalized canonical_name match
        for node in active_nodes:
            if self._normalize_name(node.canonical_name) == cand_norm:
                return node

        cand_tokens = set(cand_norm.split("_"))
        cand_attr_tokens = cand_tokens & DISTINCT_ATTRIBUTE_TOKENS
        cand_emb = compute_text_embedding(cand_norm)

        best_match: Optional[ConceptNode] = None
        best_score: float = 0.0

        for node in active_nodes:
            node_norm = self._normalize_name(node.canonical_name)
            if not node_norm:
                continue

            node_tokens = set(node_norm.split("_"))
            node_attr_tokens = node_tokens & DISTINCT_ATTRIBUTE_TOKENS

            # Do not merge attributes with differing attribute qualifiers (e.g. database vs database_host, database_host vs database_port)
            if cand_attr_tokens != node_attr_tokens:
                continue

            node_emb = compute_text_embedding(node_norm)
            sim = cosine_similarity(cand_emb, node_emb)

            shared_tokens = cand_tokens & node_tokens
            min_len = max(1, min(len(cand_tokens), len(node_tokens)))
            token_overlap = len(shared_tokens) / float(min_len)

            substring_match = (
                len(cand_norm) >= 4
                and len(node_norm) >= 4
                and (cand_norm in node_norm or node_norm in cand_norm)
            )

            score = sim
            if substring_match:
                score = max(score, self.similarity_threshold + 0.05)
            if token_overlap >= 0.75 and shared_tokens:
                score = max(score, self.similarity_threshold + 0.02)

            if score >= self.similarity_threshold and score > best_score:
                best_score = score
                best_match = node

        return best_match

    @classmethod
    def _values_equivalent(cls, val_a: str, val_b: str) -> bool:
        norm_a = cls._normalize_value(val_a).lower()
        norm_b = cls._normalize_value(val_b).lower()
        if norm_a == norm_b:
            return True
        if not norm_a or not norm_b:
            return False
        # Avoid treating distinct numbers or identifiers as equivalent
        nums_a = re.findall(r"\d+(?:\.\d+)?", norm_a)
        nums_b = re.findall(r"\d+(?:\.\d+)?", norm_b)
        if nums_a != nums_b:
            return False
        emb_a = compute_text_embedding(norm_a)
        emb_b = compute_text_embedding(norm_b)
        return cosine_similarity(emb_a, emb_b) >= 0.94

    @staticmethod
    def _infer_provenance(turn: Dict[str, Any], text: str) -> str:
        explicit = turn.get("provenance_type")
        if explicit and isinstance(explicit, str):
            return explicit
        role = str(turn.get("role") or turn.get("source") or "").strip().lower()
        if role in ("operator", "operator_crud", "crud"):
            return "operator_crud"
        if role in ("user", "user_explicit", "human"):
            return "user_explicit"
        if role in ("verified_test", "test", "pytest", "ci"):
            return "verified_test"
        lower_text = text.lower()
        if ("pytest" in lower_text or "test suite" in lower_text) and (
            "passed" in lower_text or "100%" in lower_text
        ):
            return "verified_test"
        if role in ("model_inference", "inference"):
            return "model_inference"
        return "agent_turn"

    def _create_concept_in_lake(
        self,
        lake: Any,
        candidate: ConceptNode,
        provenance_type: str,
        turn_id: int,
        timestamp: float,
    ) -> ConceptNode:
        fn = getattr(lake, "create_concept")
        sig = inspect.signature(fn)
        params = sig.parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        param_names = [
            name
            for name, p in params.items()
            if name != "self" and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]

        if "canonical_name" in param_names:
            candidate_kwargs: Dict[str, Any] = {
                "node_id": candidate.node_id,
                "canonical_name": candidate.canonical_name,
                "category": candidate.category,
                "active_value": candidate.active_value,
                "value": candidate.active_value,
                "summary": candidate.summary,
                "confidence": candidate.confidence,
                "pinned": candidate.pinned,
                "embedding": candidate.embedding,
                "source_agent_id": candidate.source_agent_id,
                "agent_id": candidate.source_agent_id,
                "provenance_type": provenance_type,
                "turn_id": turn_id,
                "created_at": timestamp,
                "updated_at": timestamp,
                "timestamp": timestamp,
            }
            filtered = {
                k: v for k, v in candidate_kwargs.items() if has_var_kw or k in param_names
            }
            res = fn(**filtered)
        else:
            extra_kwargs: Dict[str, Any] = {}
            for k, v in (
                ("provenance_type", provenance_type),
                ("turn_id", turn_id),
                ("timestamp", timestamp),
            ):
                if k in param_names:
                    extra_kwargs[k] = v
            res = fn(candidate, **extra_kwargs)

        return res if isinstance(res, ConceptNode) else candidate

    def _update_concept_in_lake(
        self,
        lake: Any,
        node_id: str,
        candidate: ConceptNode,
        provenance_type: str,
        turn_id: int,
        timestamp: float,
    ) -> Any:
        fn = getattr(lake, "update_concept")
        sig = inspect.signature(fn)
        params = sig.parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        param_names = [
            name
            for name, p in params.items()
            if name != "self" and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]

        if "node" in param_names and "node_id" not in param_names:
            candidate.node_id = node_id
            candidate.updated_at = timestamp
            return fn(candidate)

        update_kwargs: Dict[str, Any] = {}
        if "node_id" in param_names or has_var_kw:
            update_kwargs["node_id"] = node_id

        if "active_value" in param_names:
            update_kwargs["active_value"] = candidate.active_value
        elif "new_value" in param_names:
            update_kwargs["new_value"] = candidate.active_value
        elif "value" in param_names:
            update_kwargs["value"] = candidate.active_value
        elif has_var_kw:
            update_kwargs["active_value"] = candidate.active_value

        optional_fields = {
            "summary": candidate.summary,
            "category": candidate.category,
            "confidence": candidate.confidence,
            "pinned": candidate.pinned,
            "embedding": candidate.embedding,
            "source_agent_id": candidate.source_agent_id,
            "agent_id": candidate.source_agent_id,
            "provenance_type": provenance_type,
            "turn_id": turn_id,
            "updated_at": timestamp,
            "valid_from": timestamp,
            "timestamp": timestamp,
        }
        for k, v in optional_fields.items():
            if k in param_names or has_var_kw:
                update_kwargs[k] = v

        if update_kwargs and "node_id" in update_kwargs:
            return fn(**update_kwargs)
        return fn(node_id, candidate.active_value)

    @staticmethod
    def _add_edge_in_lake(lake: Any, edge: RelationEdge) -> Any:
        if not hasattr(lake, "add_edge"):
            return None
        fn = getattr(lake, "add_edge")
        sig = inspect.signature(fn)
        params = sig.parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        param_names = [
            name
            for name, p in params.items()
            if name != "self" and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]

        if "source_id" in param_names and "target_id" in param_names:
            edge_kwargs = {
                "edge_id": edge.edge_id,
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "relation_type": edge.relation_type,
                "weight": edge.weight,
                "source_agent_id": edge.source_agent_id,
                "valid_from": edge.valid_from,
                "valid_to": edge.valid_to,
            }
            filtered = {k: v for k, v in edge_kwargs.items() if has_var_kw or k in param_names}
            return fn(**filtered)
        return fn(edge)

    def ingest_turns(
        self,
        lake: Any,
        session: AgentSession,
        turns: List[Dict[str, Any]],
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Incrementally process new turns for an AgentSession using turn watermarking:
          - Filters turns with turn_index > session.last_processed_turn
          - Deduplicates extracted ConceptNodes against lake.list_concepts()
          - Resolves conflicting value updates via ConsensusEngine and records ConflictResolutionEvents
          - Updates session.last_processed_turn and session.used_tokens and persists via lake.upsert_agent_session
        """
        base_ts = float(timestamp) if timestamp is not None else time.time()
        processed_turns = 0
        skipped_turns = 0
        created_nodes = 0
        merged_nodes = 0
        conflicts_resolved = 0
        created_edges = 0

        max_turn_index = session.last_processed_turn
        added_tokens = 0

        for idx, turn in enumerate(turns):
            turn_index = int(turn.get("turn_id") or turn.get("step_index") or (idx + 1))
            if turn_index <= session.last_processed_turn:
                skipped_turns += 1
                continue

            processed_turns += 1
            if turn_index > max_turn_index:
                max_turn_index = turn_index

            text = str(turn.get("content") or turn.get("text") or turn.get("message") or "")
            added_tokens += estimate_tokens(text)

            turn_ts = float(turn.get("timestamp") or (base_ts + idx * 0.001))
            source_agent_id = str(
                turn.get("agent_id") or turn.get("source_agent_id") or session.agent_id
            )
            provenance_type = self._infer_provenance(turn, text)

            candidate_nodes, candidate_edges = self.extract_from_text(
                text=text,
                source_agent_id=source_agent_id,
                provenance_type=provenance_type,
                turn_id=turn_index,
                timestamp=turn_ts,
            )

            # Map local temporary node_id -> persisted lake node_id for edge wiring
            local_to_lake_node_id: Dict[str, str] = {}

            for candidate in candidate_nodes:
                existing_concepts = list(lake.list_concepts())
                match = self.find_matching_concept(
                    candidate.canonical_name,
                    candidate.active_value,
                    existing_concepts,
                )

                if match is None:
                    created = self._create_concept_in_lake(
                        lake=lake,
                        candidate=candidate,
                        provenance_type=provenance_type,
                        turn_id=turn_index,
                        timestamp=turn_ts,
                    )
                    lake_node_id = created.node_id
                    local_to_lake_node_id[candidate.node_id] = lake_node_id
                    self._node_provenance[lake_node_id] = provenance_type
                    val_key = (lake_node_id, self._normalize_value(candidate.active_value).lower())
                    self._value_voters.setdefault(val_key, set()).add(source_agent_id)
                    created_nodes += 1
                elif self._values_equivalent(match.active_value, candidate.active_value):
                    local_to_lake_node_id[candidate.node_id] = match.node_id
                    val_key = (match.node_id, self._normalize_value(match.active_value).lower())
                    self._value_voters.setdefault(val_key, set()).add(source_agent_id)
                    merged_nodes += 1
                else:
                    local_to_lake_node_id[candidate.node_id] = match.node_id
                    history: List[BeliefRevision] = (
                        list(lake.get_belief_history(match.node_id))
                        if hasattr(lake, "get_belief_history")
                        else []
                    )
                    active_rev: Optional[BeliefRevision] = None
                    for rev in reversed(history):
                        if rev.is_active:
                            active_rev = rev
                            break
                    if active_rev is None and history:
                        active_rev = history[-1]

                    if active_rev is None:
                        active_rev = BeliefRevision(
                            revision_id=f"rev_{match.node_id}_init",
                            node_id=match.node_id,
                            value=match.active_value,
                            confidence=match.confidence,
                            source_agent_id=match.source_agent_id,
                            provenance_type=self._node_provenance.get(match.node_id, "agent_turn"),
                            turn_id=0,
                            valid_from=match.updated_at or match.created_at or 0.0,
                        )

                    candidate_rev = BeliefRevision(
                        revision_id=f"rev_{match.node_id}_{turn_index}_{uuid.uuid4().hex[:6]}",
                        node_id=match.node_id,
                        value=candidate.active_value,
                        confidence=candidate.confidence,
                        source_agent_id=source_agent_id,
                        provenance_type=provenance_type,
                        turn_id=turn_index,
                        valid_from=turn_ts,
                    )

                    existing_val_key = (
                        match.node_id,
                        self._normalize_value(active_rev.value).lower(),
                    )
                    cand_val_key = (
                        match.node_id,
                        self._normalize_value(candidate.active_value).lower(),
                    )
                    self._value_voters.setdefault(existing_val_key, {active_rev.source_agent_id})
                    self._value_voters.setdefault(cand_val_key, set()).add(source_agent_id)

                    # Also count historical supporting revisions from get_belief_history
                    for hist_rev in history:
                        h_key = (match.node_id, self._normalize_value(hist_rev.value).lower())
                        self._value_voters.setdefault(h_key, set()).add(hist_rev.source_agent_id)

                    vote_counts: Dict[str, int] = {
                        active_rev.revision_id: len(self._value_voters.get(existing_val_key, {1})),
                        candidate_rev.revision_id: len(self._value_voters.get(cand_val_key, {1})),
                        active_rev.value: len(self._value_voters.get(existing_val_key, {1})),
                        candidate_rev.value: len(self._value_voters.get(cand_val_key, {1})),
                    }
                    if isinstance(turn.get("vote_counts"), dict):
                        vote_counts.update(turn["vote_counts"])

                    winner, _loser, event = self.consensus_engine.resolve(
                        existing_revision=active_rev,
                        candidate_revision=candidate_rev,
                        vote_counts=vote_counts,
                        resolved_at=turn_ts,
                    )
                    if hasattr(lake, "record_conflict_event"):
                        lake.record_conflict_event(event)
                    conflicts_resolved += 1

                    if winner.revision_id == candidate_rev.revision_id:
                        self._update_concept_in_lake(
                            lake=lake,
                            node_id=match.node_id,
                            candidate=candidate,
                            provenance_type=provenance_type,
                            turn_id=turn_index,
                            timestamp=turn_ts,
                        )
                        self._node_provenance[match.node_id] = provenance_type

            for edge in candidate_edges:
                mapped_src = local_to_lake_node_id.get(edge.source_id, edge.source_id)
                mapped_tgt = local_to_lake_node_id.get(edge.target_id, edge.target_id)
                if mapped_src != mapped_tgt:
                    edge.source_id = mapped_src
                    edge.target_id = mapped_tgt
                    self._add_edge_in_lake(lake, edge)
                    created_edges += 1

        if processed_turns > 0:
            session.last_processed_turn = max_turn_index
            session.used_tokens += added_tokens
            session.updated_at = base_ts
            if hasattr(lake, "upsert_agent_session"):
                lake.upsert_agent_session(session)

        return {
            "processed_turns": processed_turns,
            "skipped_turns": skipped_turns,
            "created_nodes": created_nodes,
            "merged_nodes": merged_nodes,
            "conflicts_resolved": conflicts_resolved,
            "created_edges": created_edges,
        }
