"""
Conversational Engine for Federated Context Lake (context-palace).
Provides multi-turn conversational Q&A, natural language and slash command intent classification,
declarative memory CRUD (/remember, /forget, /update, /list), bitemporal time-travel queries (/as-of),
model context window inspection (/windows), interactive REPL, and deterministic offline synthesis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

from context_palace.consensus import ConsensusEngine
from context_palace.lake import ContextLake
from context_palace.model_registry import ModelContextRegistry
from context_palace.models import (
    AgentSession,
    BeliefRevision,
    ConceptNode,
    ConflictResolutionEvent,
    RelationEdge,
    compute_text_embedding,
    estimate_tokens,
)
from context_palace.retrieval import DualDirectionRetriever

# Alias for ConceptNode in memory palace conversations
MemoryNode = ConceptNode


@dataclass
class ConversationTurn:
    """Represents a single conversational turn (user or assistant) in Context Palace."""

    turn_id: str
    role: str  # "user" or "assistant"
    message: str
    intent: str  # "query", "remember", "forget", "update", "list", "time_travel", "window_status", "conflict_resolve"
    cited_node_ids: List[str] = field(default_factory=list)
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConversationalMemoryPalace:
    """
    Conversational engine over the Federated Context Lake.
    Supports multi-turn interactive chat, declarative CRUD intents,
    bitemporal time-travel recall, context headroom tracking, and offline deterministic synthesis.
    """

    def __init__(
        self,
        lake: ContextLake,
        retriever: Optional[DualDirectionRetriever] = None,
        consensus: Optional[ConsensusEngine] = None,
        registry: Optional[ModelContextRegistry] = None,
        session_id: Optional[str] = None,
        agent_name: str = "context-palace-chat",
        model_id: str = "gemini-2.5-pro",
        persist_turns: bool = True,
        llm_callback: Optional[Callable[[str, List[Dict[str, Any]]], str]] = None,
    ) -> None:
        self.lake = lake
        self.retriever = retriever if retriever is not None else DualDirectionRetriever(lake)
        self.consensus = consensus if consensus is not None else ConsensusEngine()
        self.registry = registry if registry is not None else ModelContextRegistry()
        self.session_id = session_id or f"session-{uuid.uuid4().hex[:8]}"
        self.agent_name = agent_name
        self.model_id = model_id
        self.persist_turns = persist_turns
        self.llm_callback = llm_callback
        self.history: List[ConversationTurn] = []

        self._ensure_interface_compatibility()

        # Only register an agent_sessions row when an explicit session_id and persist_turns=True are requested
        if session_id is not None and self.persist_turns:
            existing_session = self.lake.get_agent_session(self.session_id)
            if existing_session is None:
                spec = self.registry.resolve_model_window(self.model_id)
                new_session = AgentSession(
                    agent_id=self.session_id,
                    harness_type="conversation",
                    model_id=self.model_id,
                    max_context_tokens=spec.max_context_tokens,
                    used_tokens=0,
                    last_processed_turn=0,
                    updated_at=time.time(),
                    metadata={"agent_name": self.agent_name},
                )
                self.lake.upsert_agent_session(new_session)

    def _ensure_interface_compatibility(self) -> None:
        """Dynamically attach bridge helpers if missing on collaborator instances."""
        # 1. lake.insert_node
        if not hasattr(self.lake, "insert_node") and hasattr(self.lake, "create_concept"):
            def _insert_node(node: ConceptNode) -> ConceptNode:
                return self.lake.create_concept(
                    canonical_name=node.canonical_name,
                    category=node.category,
                    value=node.active_value,
                    summary=node.summary,
                    confidence=node.confidence,
                    source_agent_id=node.source_agent_id,
                    node_id=node.node_id,
                )
            setattr(self.lake, "insert_node", _insert_node)

        # 2. lake.get_active_nodes
        if not hasattr(self.lake, "get_active_nodes"):
            def _get_active_nodes(as_of: Optional[float] = None) -> List[ConceptNode]:
                if as_of is not None and hasattr(self.lake, "snapshot"):
                    snap = self.lake.snapshot(as_of=as_of)
                    if isinstance(snap, dict):
                        return snap.get("nodes", snap.get("concepts", []))
                if hasattr(self.lake, "list_concepts"):
                    return self.lake.list_concepts(include_tombstoned=False)
                return []
            setattr(self.lake, "get_active_nodes", _get_active_nodes)

        # 3. lake.get_subgraph
        if not hasattr(self.lake, "get_subgraph"):
            def _get_subgraph(query: str = "", top_k: int = 8, hops: int = 1) -> Dict[str, Any]:
                nodes, edges = self.retriever.query_subgraph(query=query, top_k=top_k, hops=hops)
                return {
                    "nodes": [n.to_dict() if hasattr(n, "to_dict") else n for n in nodes],
                    "edges": [e.to_dict() if hasattr(e, "to_dict") else e for e in edges],
                }
            setattr(self.lake, "get_subgraph", _get_subgraph)

        # 4. lake.supersede_node
        if not hasattr(self.lake, "supersede_node") and hasattr(self.lake, "update_concept"):
            def _supersede_node(old_node_id: str, new_node: ConceptNode) -> ConceptNode:
                return self.lake.update_concept(
                    node_id_or_name=old_node_id,
                    new_value=new_node.active_value,
                    summary=new_node.summary,
                    confidence=new_node.confidence,
                    source_agent_id=new_node.source_agent_id,
                )
            setattr(self.lake, "supersede_node", _supersede_node)

        # 5. retriever.retrieve
        if not hasattr(self.retriever, "retrieve") and hasattr(self.retriever, "query_subgraph"):
            def _retrieve(query: str, top_k: int = 8) -> Dict[str, Any]:
                nodes, edges = self.retriever.query_subgraph(query=query, top_k=top_k, hops=1)
                return {
                    "nodes": nodes,
                    "edges": edges,
                }
            setattr(self.retriever, "retrieve", _retrieve)

        # 6. consensus.upsert_with_consensus
        if not hasattr(self.consensus, "upsert_with_consensus"):
            def _upsert_with_consensus(lake: Any, node: ConceptNode) -> ConceptNode:
                if hasattr(lake, "insert_node"):
                    return lake.insert_node(node)
                return lake.create_concept(
                    canonical_name=node.canonical_name,
                    category=node.category,
                    value=node.active_value,
                    summary=node.summary,
                    confidence=node.confidence,
                    source_agent_id=node.source_agent_id,
                    node_id=node.node_id,
                )
            setattr(self.consensus, "upsert_with_consensus", _upsert_with_consensus)

        # 7. consensus.resolve_conflict
        if not hasattr(self.consensus, "resolve_conflict"):
            def _resolve_conflict(
                old_node: ConceptNode,
                new_node: ConceptNode,
                resolved_at: Optional[float] = None,
            ) -> Tuple[BeliefRevision, BeliefRevision, ConflictResolutionEvent]:
                now = resolved_at if resolved_at is not None else time.time()
                old_rev = BeliefRevision(
                    revision_id=f"rev_{old_node.node_id[:8]}",
                    node_id=old_node.node_id,
                    value=old_node.active_value,
                    confidence=old_node.confidence,
                    source_agent_id=old_node.source_agent_id,
                    valid_from=old_node.updated_at or (now - 100.0),
                )
                new_rev = BeliefRevision(
                    revision_id=f"rev_{new_node.node_id[:8]}",
                    node_id=new_node.node_id,
                    value=new_node.active_value,
                    confidence=new_node.confidence,
                    source_agent_id=new_node.source_agent_id,
                    valid_from=now,
                )
                return self.consensus.resolve(old_rev, new_rev, resolved_at=now)
            setattr(self.consensus, "resolve_conflict", _resolve_conflict)

    # -------------------------------------------------------------------------
    # Intent Classification
    # -------------------------------------------------------------------------

    def classify_intent(self, message: str) -> Tuple[str, Dict[str, Any]]:
        """
        Deterministically parses natural language and slash commands into intents:
        - /remember <fact> or "remember that <fact>" / "save that <fact>" / "note that <fact>"
        - /forget <target> or "forget <target>" / "delete node <node_id>" / "remove memory about <topic>"
        - /update <node_id> to <new_content> or "update <topic> to <new_content>" / "change <topic> to <new_content>"
        - /list or "list everything" / "show all memories" / "what is in the lake"
        - /as-of <timestamp> <query> or "as of <timestamp>" / "what did we believe at <timestamp>"
        - /windows or "context window" / "model limits" / "token budget"
        - Default -> "query"
        """
        raw = (message or "").strip()
        lower = raw.lower()

        # 1. Remember intent
        remember_match = re.match(
            r"^(?:/remember|/save|/note|(?:please\s+)?remember(?:\s+that)?|(?:please\s+)?save(?:\s+that)?|(?:please\s+)?note(?:\s+that)?)\b\s*(.*)",
            raw,
            flags=re.IGNORECASE,
        )
        if remember_match and remember_match.group(1).strip():
            content = remember_match.group(1).strip()
            memory_type = self._extract_memory_type(content)
            # Clean content if bracketed tag was used
            content = re.sub(r"^\[[a-zA-Z0-9_-]+\]\s*", "", content).strip()
            content = re.sub(r"--(?:type|category)\s+[a-zA-Z0-9_-]+", "", content).strip()
            return "remember", {
                "content": content,
                "memory_type": memory_type,
                "raw": raw,
            }

        # 2. Forget intent
        forget_match = re.match(
            r"^(?:/forget|/delete|/remove|(?:please\s+)?forget|(?:please\s+)?delete(?:\s+node)?|(?:please\s+)?remove(?:\s+memory(?:\s+about)?)?)\b\s*(.*)",
            raw,
            flags=re.IGNORECASE,
        )
        if forget_match and forget_match.group(1).strip():
            target = forget_match.group(1).strip()
            target = re.sub(r"^(?:node\s+|about\s+|memory\s+about\s+)", "", target, flags=re.IGNORECASE).strip()
            # If target was empty or purely "node", skip
            if target:
                return "forget", {
                    "target": target,
                    "raw": raw,
                }

        # 3. Update intent
        update_match = re.match(
            r"^(?:/update|(?:please\s+)?update|(?:please\s+)?change)\b\s+(.*?)\s+(?:to|with|=|->|into)\s+(.*)",
            raw,
            flags=re.IGNORECASE,
        )
        if update_match:
            target = update_match.group(1).strip()
            target = re.sub(r"^(?:node\s+)", "", target, flags=re.IGNORECASE).strip()
            new_content = update_match.group(2).strip()
            if target and new_content:
                return "update", {
                    "target": target,
                    "new_content": new_content,
                    "raw": raw,
                }

        # 4. List intent
        list_patterns = [
            r"^/list\b",
            r"^/all$",
            r"^/memories$",
            r"^(?:please\s+)?list(?:\s+(?:everything|all(?:\s+memories)?|memories))?$",
            r"^(?:please\s+)?show(?:\s+all)?(?:\s+memories|\s+lake)?$",
            r"^what\s+(?:is|'s)\s+in\s+(?:the\s+lake|memory)\??$",
        ]
        if any(re.match(p, lower) for p in list_patterns):
            return "list", {"raw": raw}

        # 5. Time Travel intent (/as-of or as of ...)
        as_of_match = re.match(
            r"^(?:/as-of|/time-travel|as\s+of|snapshot\s+as\s+of|what\s+did\s+we\s+believe\s+at)\b\s*([^\s,]+)?(?:\s*,\s*|\s+)?(.*)",
            raw,
            flags=re.IGNORECASE,
        )
        if as_of_match:
            as_of_str = (as_of_match.group(1) or "").strip()
            query_str = (as_of_match.group(2) or "").strip()
            query_str = re.sub(r"^(?:about\s+|regarding\s+)", "", query_str, flags=re.IGNORECASE).strip()
            if as_of_str:
                return "time_travel", {
                    "as_of": as_of_str,
                    "query": query_str,
                    "raw": raw,
                }

        # 6. Window Status intent
        window_patterns = [
            r"^/windows?$",
            r"^/budget$",
            r"^(?:show\s+)?context\s+windows?$",
            r"^(?:show\s+)?model\s+limits?$",
            r"^(?:show\s+)?token\s+budgets?$",
            r"^window\s+status$",
        ]
        if any(re.match(p, lower) for p in window_patterns):
            return "window_status", {"raw": raw}

        # 7. Conflict Resolve intent
        conflict_match = re.match(
            r"^(?:/resolve|/conflict|resolve(?:\s+conflict)?)\b\s*(.*)",
            raw,
            flags=re.IGNORECASE,
        )
        if conflict_match and conflict_match.group(1).strip():
            return "conflict_resolve", {
                "target": conflict_match.group(1).strip(),
                "raw": raw,
            }

        # 8. Default: Query intent
        return "query", {
            "query": raw,
            "raw": raw,
        }

    @staticmethod
    def _extract_memory_type(text: str) -> str:
        """Extract memory category from explicit flag, tag, or semantic keyword heuristic."""
        # 1. Explicit flag e.g. --type preference or --category architecture
        flag_match = re.search(r"--(?:type|category)\s+([a-zA-Z0-9_-]+)", text, flags=re.IGNORECASE)
        if flag_match:
            return flag_match.group(1).lower()

        # 2. Bracketed tag e.g. [preference] ...
        tag_match = re.match(r"^\[([a-zA-Z0-9_-]+)\]", text)
        if tag_match:
            return tag_match.group(1).lower()

        # 3. Colon prefix e.g. "preference: user prefers dark mode"
        colon_match = re.match(r"^([a-zA-Z0-9_-]+)\s*:", text)
        if colon_match:
            prefix = colon_match.group(1).lower()
            if prefix in {"preference", "decision", "architecture", "constraint", "config", "fact", "entity", "task"}:
                return prefix

        # 4. Semantic heuristics
        t_lower = text.lower()
        if any(w in t_lower for w in ("prefer", "preference", "likes", "dislikes", "favorite")):
            return "preference"
        if any(w in t_lower for w in ("decid", "decision", "agreed", "chose", "selected")):
            return "decision"
        if any(w in t_lower for w in ("architect", "pattern", "service", "microservice", "monorepo", "database", "backend")):
            return "architecture"
        if any(w in t_lower for w in ("constraint", "must not", "never", "limit", "restriction", "prohibited")):
            return "constraint"
        if any(w in t_lower for w in ("config", "setting", "env", "port", "url", "parameter")):
            return "config"
        return "fact"

    @staticmethod
    def _derive_canonical_name(content: str) -> str:
        """Derive a concise canonical name for a concept from memory content."""
        cleaned = content.strip()
        if ":" in cleaned:
            candidate = cleaned.split(":", 1)[0].strip()
            if len(candidate.split()) <= 6:
                return candidate

        words = cleaned.split()
        if len(words) <= 5:
            return cleaned.rstrip(".,;:")
        return " ".join(words[:4]).rstrip(".,;:")

    @staticmethod
    def _parse_timestamp(ts: Any) -> float:
        """Convert ISO timestamp string, numeric string, or epoch float to float timestamp."""
        if isinstance(ts, (int, float)):
            return float(ts)
        if not ts:
            return time.time()
        s = str(ts).strip()
        try:
            return float(s)
        except ValueError:
            pass
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            return time.time()

    # -------------------------------------------------------------------------
    # Core Conversational Execution (ask)
    # -------------------------------------------------------------------------

    def ask(self, message: str, as_of: Optional[str] = None, top_k: int = 8) -> Dict[str, Any]:
        """
        Executes the classified intent against self.lake, self.retriever, self.consensus, and self.registry.
        Appends ConversationTurns to self.history, tracks token headroom, and returns a structured response dict.
        """
        intent, params = self.classify_intent(message)
        now_ts = time.time()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Allow explicit as_of argument to override intent as_of
        effective_as_of = as_of if as_of is not None else params.get("as_of")
        if effective_as_of is not None and intent != "time_travel" and intent not in ("remember", "forget", "update"):
            intent = "time_travel"
            params["as_of"] = effective_as_of
            params["query"] = message

        answer = ""
        cited_nodes: List[Dict[str, Any]] = []
        cited_node_ids: List[str] = []
        subgraph: Dict[str, Any] = {"nodes": [], "edges": []}

        # ---------------------------------------------------------------------
        # 1. Intent: remember
        # ---------------------------------------------------------------------
        if intent == "remember":
            content = params.get("content", "").strip()
            mem_type = params.get("memory_type", "fact")
            kv_match = re.match(r"^([a-zA-Z0-9_]+)\s+(?:is|=)\s+(.+)$", content, flags=re.IGNORECASE)
            if kv_match:
                canonical_name = kv_match.group(1).strip()
                active_val = kv_match.group(2).strip()
            else:
                canonical_name = self._derive_canonical_name(content)
                active_val = content

            new_node = ConceptNode(
                node_id=f"node-{uuid.uuid4().hex[:10]}",
                canonical_name=canonical_name,
                category=mem_type,
                active_value=active_val,
                summary=f"{canonical_name}: {active_val[:120]}",
                confidence=0.95,
                pinned=False,
                embedding=compute_text_embedding(f"{canonical_name} {mem_type} {active_val}"),
                source_agent_id=self.agent_name,
                created_at=now_ts,
                updated_at=now_ts,
                tombstoned=False,
            )

            # Insert via consensus / lake
            if hasattr(self.consensus, "upsert_with_consensus"):
                saved_node = self.consensus.upsert_with_consensus(self.lake, new_node)
            elif hasattr(self.lake, "insert_node"):
                saved_node = self.lake.insert_node(new_node)
            else:
                saved_node = self.lake.create_concept(
                    canonical_name=new_node.canonical_name,
                    category=new_node.category,
                    value=new_node.active_value,
                    summary=new_node.summary,
                    confidence=new_node.confidence,
                    source_agent_id=new_node.source_agent_id,
                    provenance_type="user_explicit",
                    timestamp=now_ts,
                    node_id=new_node.node_id,
                )

            # Discover related existing nodes and link them
            related_nodes, _ = self.retriever.query_subgraph(query=content, top_k=3, hops=0)
            related_nodes = [
                n for n in related_nodes
                if getattr(n, "node_id", "") != saved_node.node_id and not getattr(n, "tombstoned", False)
            ]

            created_edges: List[RelationEdge] = []
            for rel in related_nodes:
                try:
                    edge = self.lake.add_edge(
                        source_id=saved_node.node_id,
                        target_id=rel.node_id,
                        relation_type="relates_to",
                        weight=0.85,
                        source_agent_id=self.agent_name,
                        timestamp=now_ts,
                    )
                    created_edges.append(edge)
                except Exception:
                    pass

            cited_nodes = [saved_node.to_dict()] + [r.to_dict() for r in related_nodes]
            cited_node_ids = [saved_node.node_id] + [r.node_id for r in related_nodes]
            subgraph = {
                "nodes": cited_nodes,
                "edges": [e.to_dict() for e in created_edges],
            }

            link_note = f" Linked with {len(created_edges)} related memories." if created_edges else ""
            answer = (
                f"I have committed this memory to the palace: [{saved_node.node_id[:8]}] "
                f"({saved_node.canonical_name}): \"{saved_node.active_value}\" "
                f"[category: {saved_node.category}, confidence: {saved_node.confidence:.2f}].{link_note}"
            )

        # ---------------------------------------------------------------------
        # 2. Intent: forget
        # ---------------------------------------------------------------------
        elif intent == "forget":
            target = params.get("target", "").strip()
            all_active = self.lake.list_concepts(include_tombstoned=False)
            matched_nodes: List[ConceptNode] = []

            # 1. Exact ID match or prefix match
            for n in all_active:
                if n.node_id == target or n.node_id.startswith(target) or target.startswith(n.node_id):
                    matched_nodes.append(n)

            # 2. Canonical name exact / case-insensitive
            if not matched_nodes:
                for n in all_active:
                    if n.canonical_name.lower() == target.lower():
                        matched_nodes.append(n)

            # 3. Substring match in canonical_name or active_value
            if not matched_nodes:
                for n in all_active:
                    if target.lower() in n.canonical_name.lower() or target.lower() in n.active_value.lower():
                        matched_nodes.append(n)

            # 4. Top semantic retrieval match
            if not matched_nodes:
                cand_nodes, _ = self.retriever.query_subgraph(query=target, top_k=2, hops=0)
                cand_nodes = [c for c in cand_nodes if not c.tombstoned]
                if cand_nodes:
                    matched_nodes.append(cand_nodes[0])

            if not matched_nodes:
                answer = f"No active memory found matching '{target}'. Use /list to see all active memories."
            else:
                forgotten_ids = []
                for node in matched_nodes:
                    self.lake.delete_concept(node.node_id, hard=False, timestamp=now_ts)
                    forgotten_ids.append(node.node_id)

                cited_nodes = [n.to_dict() for n in matched_nodes]
                cited_node_ids = forgotten_ids
                subgraph = {"nodes": cited_nodes, "edges": []}

                node_labels = ", ".join(f"[{n.node_id[:8]}] ({n.canonical_name})" for n in matched_nodes)
                answer = (
                    f"I have removed the memory from active palace recall: {node_labels}. "
                    f"Belief revision closed and marked invalid (valid_until set to {now_iso})."
                )

        # ---------------------------------------------------------------------
        # 3. Intent: update
        # ---------------------------------------------------------------------
        elif intent == "update":
            target = params.get("target", "").strip()
            new_content = params.get("new_content", "").strip()

            all_active = self.lake.list_concepts(include_tombstoned=False)
            old_node: Optional[ConceptNode] = None

            # Match target
            for n in all_active:
                if n.node_id == target or n.node_id.startswith(target) or target.startswith(n.node_id):
                    old_node = n
                    break
            if old_node is None:
                for n in all_active:
                    if n.canonical_name.lower() == target.lower():
                        old_node = n
                        break
            if old_node is None:
                for n in all_active:
                    if target.lower() in n.canonical_name.lower():
                        old_node = n
                        break
            if old_node is None:
                cands, _ = self.retriever.query_subgraph(query=target, top_k=1, hops=0)
                cands = [c for c in cands if not c.tombstoned]
                if cands:
                    old_node = cands[0]

            if old_node is None:
                answer = f"Target memory '{target}' not found to update. Use /remember to store it as a new memory."
            else:
                # Create a new superseding concept node
                new_cat = self._extract_memory_type(new_content) if self._extract_memory_type(new_content) != "fact" else old_node.category
                new_node = ConceptNode(
                    node_id=f"node-{uuid.uuid4().hex[:10]}",
                    canonical_name=old_node.canonical_name,
                    category=new_cat,
                    active_value=new_content,
                    summary=f"{old_node.canonical_name}: {new_content[:120]}",
                    confidence=0.95,
                    pinned=old_node.pinned,
                    embedding=compute_text_embedding(f"{old_node.canonical_name} {new_content}"),
                    source_agent_id=self.agent_name,
                    created_at=now_ts,
                    updated_at=now_ts,
                    tombstoned=False,
                )

                # Insert new node
                if hasattr(self.lake, "insert_node"):
                    saved_new_node = self.lake.insert_node(new_node)
                else:
                    saved_new_node = self.lake.create_concept(
                        canonical_name=new_node.canonical_name,
                        category=new_node.category,
                        value=new_node.active_value,
                        summary=new_node.summary,
                        confidence=new_node.confidence,
                        source_agent_id=new_node.source_agent_id,
                        provenance_type="user_explicit",
                        timestamp=now_ts,
                        node_id=new_node.node_id,
                    )

                # Invalidate / soft-delete old node first so previous relations are closed
                self.lake.delete_concept(old_node.node_id, hard=False, timestamp=now_ts)

                # Add explicit active SUPERSEDES relation edge
                edge = self.lake.add_edge(
                    source_id=saved_new_node.node_id,
                    target_id=old_node.node_id,
                    relation_type="SUPERSEDES",
                    weight=1.0,
                    source_agent_id=self.agent_name,
                    timestamp=now_ts,
                )

                # Record consensus conflict resolution audit event
                if hasattr(self.consensus, "resolve_conflict"):
                    try:
                        _, _, cre = self.consensus.resolve_conflict(old_node, saved_new_node, resolved_at=now_ts)
                        self.lake.record_conflict_event(cre)
                    except Exception:
                        pass

                cited_nodes = [saved_new_node.to_dict(), old_node.to_dict()]
                cited_node_ids = [saved_new_node.node_id, old_node.node_id]
                subgraph = {
                    "nodes": cited_nodes,
                    "edges": [edge.to_dict()],
                }

                answer = (
                    f"Updated memory successfully: {saved_new_node.node_id} --[SUPERSEDES]--> {old_node.node_id}\n"
                    f"New active belief for '{saved_new_node.canonical_name}': \"{saved_new_node.active_value}\" "
                    f"[{saved_new_node.category}, confidence: {saved_new_node.confidence:.2f}]."
                )

        # ---------------------------------------------------------------------
        # 4. Intent: time_travel
        # ---------------------------------------------------------------------
        elif intent == "time_travel":
            as_of_val = effective_as_of or params.get("as_of", now_ts)
            as_of_ts = self._parse_timestamp(as_of_val)
            query_str = params.get("query", "").strip()

            # Retrieve active nodes as of timestamp
            if hasattr(self.lake, "get_active_nodes"):
                hist_nodes = self.lake.get_active_nodes(as_of=as_of_ts)
            elif hasattr(self.lake, "snapshot"):
                snap = self.lake.snapshot(as_of=as_of_ts)
                hist_nodes = snap.get("nodes", snap.get("concepts", [])) if isinstance(snap, dict) else []
            else:
                hist_nodes = self.lake.list_concepts(include_tombstoned=False)

            # Filter or score nodes against query_str if provided
            if query_str:
                query_vec = compute_text_embedding(query_str)
                scored = [
                    (self.retriever.score_node(n, query_str, query_vec, include_pinned=True), n)
                    for n in hist_nodes
                ]
                scored.sort(key=lambda pair: pair[0], reverse=True)
                selected_hist_nodes = [n for sc, n in scored[:top_k] if sc > 0.0] or [n for _, n in scored[:2]]
            else:
                selected_hist_nodes = hist_nodes[:top_k]

            cited_nodes = [n.to_dict() for n in selected_hist_nodes]
            cited_node_ids = [n.node_id for n in selected_hist_nodes]
            subgraph = {"nodes": cited_nodes, "edges": []}

            lines = [f"**Time-Travel Snapshot (as of `{as_of_val}`, t={as_of_ts:.1f}):**"]
            if not selected_hist_nodes:
                lines.append("No active memories existed in the federated lake at that timestamp.")
            else:
                for n in selected_hist_nodes:
                    lines.append(
                        f"• [{n.node_id[:8]}] ({n.source_agent_id}, {n.category}, {n.confidence:.2f}): "
                        f"**{n.canonical_name}** was: \"{n.active_value}\""
                    )

            # Check if active belief differs today
            current_active = self.lake.list_concepts(include_tombstoned=False)
            curr_map = {n.canonical_name.lower(): n for n in current_active}
            diffs = []
            for hn in selected_hist_nodes:
                curr = curr_map.get(hn.canonical_name.lower())
                if curr and curr.active_value != hn.active_value:
                    diffs.append(f"  - `{hn.canonical_name}`: \"{hn.active_value}\" ➔ \"{curr.active_value}\" today")
                elif not curr:
                    diffs.append(f"  - `{hn.canonical_name}`: tombstoned or superseded in current state")
            if diffs:
                lines.append("\nEvolution since that timestamp:")
                lines.extend(diffs)

            answer = "\n".join(lines)

        # ---------------------------------------------------------------------
        # 5. Intent: window_status
        # ---------------------------------------------------------------------
        elif intent == "window_status":
            session = self.lake.get_agent_session(self.session_id)
            all_sessions = self.lake.list_agent_sessions()
            spec = self.registry.resolve_model_window(self.model_id)

            max_tokens = session.max_context_tokens if session else spec.max_context_tokens
            used_tokens = session.used_tokens if session else 0
            headroom = session.remaining_headroom if session else max(0, max_tokens - used_tokens)
            util_pct = session.utilization_pct if session else (round((used_tokens / max(1, max_tokens)) * 100.0, 2))
            capsule_budget = session.adaptive_capsule_budget() if session else 1024

            lines = [
                f"### Model Context Window & Session Budget (`{self.session_id}`)",
                f"- **Model:** `{spec.canonical_name}` (`{spec.model_id}` | Provider: `{spec.provider}`)",
                f"- **Capacity:** `{max_tokens:,} tokens`",
                f"- **Used Tokens:** `{used_tokens:,} tokens` ({util_pct:.1f}% utilization)",
                f"- **Remaining Headroom:** `{headroom:,} tokens`",
                f"- **Adaptive Capsule Budget:** `{capsule_budget:,} tokens` (12% headroom ceiling)",
            ]
            if len(all_sessions) > 1:
                lines.append(f"\n**Other Connected Agent Sessions ({len(all_sessions)} total):**")
                for s in all_sessions:
                    if s.agent_id != self.session_id:
                        lines.append(
                            f"- `{s.agent_id}`: model `{s.model_id}`, `{s.used_tokens:,} / {s.max_context_tokens:,} tokens` "
                            f"({s.utilization_pct:.1f}%)"
                        )
            answer = "\n".join(lines)

        # ---------------------------------------------------------------------
        # 6. Intent: list
        # ---------------------------------------------------------------------
        elif intent == "list":
            active_nodes = self.lake.list_concepts(include_tombstoned=False)
            all_nodes = self.lake.list_concepts(include_tombstoned=True)
            tombstoned_nodes = [n for n in all_nodes if n.tombstoned]
            edges = self.lake.list_edges(active_only=False)
            sessions = self.lake.list_agent_sessions()

            lines = [f"### Federated Context Lake Overview ({len(active_nodes)} active, {len(tombstoned_nodes)} superseded)"]
            if active_nodes:
                lines.append("\n**Active Concepts:**")
                for n in active_nodes:
                    pin = "[PINNED] " if n.pinned else ""
                    lines.append(
                        f"• {pin}[{n.node_id[:8]}] **{n.canonical_name}** [`{n.category}`]: "
                        f"\"{n.active_value}\" *(conf={n.confidence:.2f}, src={n.source_agent_id})*"
                    )
            if tombstoned_nodes:
                lines.append(f"\n**Superseded / Inactive Concepts ({len(tombstoned_nodes)}):**")
                for n in tombstoned_nodes:
                    lines.append(f"• ~~[{n.node_id[:8]}] {n.canonical_name}~~: \"{n.active_value}\" *(tombstoned)*")
            if edges:
                lines.append(f"\n**Relations ({len(edges)}):**")
                for e in edges:
                    status = " (expired)" if e.valid_to else ""
                    lines.append(f"• `[{e.source_id[:8]}]` --[{e.relation_type}]--> `[{e.target_id[:8]}]`{status}")
            if sessions:
                lines.append(f"\n**Connected Agent Sessions ({len(sessions)}):**")
                for s in sessions:
                    lines.append(f"• `{s.agent_id}` [{s.model_id}]: {s.used_tokens:,} / {s.max_context_tokens:,} tokens")

            cited_nodes = [n.to_dict() for n in active_nodes]
            cited_node_ids = [n.node_id for n in active_nodes]
            subgraph = {
                "nodes": cited_nodes,
                "edges": [e.to_dict() for e in edges],
            }
            answer = "\n".join(lines)

        # ---------------------------------------------------------------------
        # 7. Intent: conflict_resolve
        # ---------------------------------------------------------------------
        elif intent == "conflict_resolve":
            target = params.get("target", "").strip()
            events = self.lake.list_conflict_events(node_id=target if target else None)
            if not events:
                answer = f"No conflict resolution audit events found for '{target}'."
            else:
                lines = [f"### Conflict Resolution Audit Log ({len(events)} events)"]
                for ev in events:
                    lines.append(
                        f"• Event `{ev.event_id}` for Node `{ev.node_id[:8]}`: "
                        f"Winner `{ev.winning_agent_id}` ({ev.winning_score:.3f}) > "
                        f"Loser `{ev.losing_agent_id}` ({ev.losing_score:.3f}) — Reason: {ev.reason}"
                    )
                answer = "\n".join(lines)

        # ---------------------------------------------------------------------
        # 8. Intent: query (Natural Language Question)
        # ---------------------------------------------------------------------
        else:
            query = params.get("query", message).strip()
            retrieved_nodes, retrieved_edges = self.retriever.query_subgraph(
                query=query,
                top_k=top_k,
                hops=1,
            )
            # Filter tombstoned nodes from active retrieval
            retrieved_nodes = [n for n in retrieved_nodes if not n.tombstoned]

            cited_nodes = [n.to_dict() for n in retrieved_nodes]
            cited_node_ids = [n.node_id for n in retrieved_nodes]
            subgraph = {
                "nodes": cited_nodes,
                "edges": [e.to_dict() for e in retrieved_edges],
            }

            if not retrieved_nodes:
                answer = (
                    f"I searched the federated memory lake for '{query}', but found no matching active memories. "
                    f"You can teach me by saying `/remember <fact>`."
                )
            elif self.llm_callback is not None:
                # Optional external LLM callback
                answer = self.llm_callback(query, cited_nodes)
            else:
                # 100% deterministic offline grounded synthesis
                top_node = retrieved_nodes[0]
                lines = [
                    f"Found {len(retrieved_nodes)} relevant belief(s) — Grounded in the federated memory lake, here is what is known regarding **{query}**:\n",
                    f"Primary fact: **{top_node.canonical_name}** is documented as: \"{top_node.active_value}\".\n",
                    "**Cited Palace Memories:**",
                ]
                for n in retrieved_nodes:
                    lines.append(
                        f"• [{n.node_id[:8]}] ({n.source_agent_id}, {n.category}, {n.confidence:.2f}): "
                        f"**{n.canonical_name}** ➔ {n.active_value}"
                    )
                if retrieved_edges:
                    lines.append("\n**Graph Relations:**")
                    for e in retrieved_edges:
                        lines.append(f"• `[{e.source_id[:8]}]` --[{e.relation_type}]--> `[{e.target_id[:8]}]` (weight={e.weight:.2f})")
                answer = "\n".join(lines)

        # ---------------------------------------------------------------------
        # Turn Recording & Token Accounting
        # ---------------------------------------------------------------------
        user_turn = ConversationTurn(
            turn_id=f"turn-{uuid.uuid4().hex[:8]}",
            role="user",
            message=message,
            intent=intent,
            cited_node_ids=[],
            timestamp=now_iso,
            metadata={"intent_params": params},
        )
        assistant_turn = ConversationTurn(
            turn_id=f"turn-{uuid.uuid4().hex[:8]}",
            role="assistant",
            message=answer,
            intent=intent,
            cited_node_ids=cited_node_ids,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={"cited_count": len(cited_nodes)},
        )

        self.history.append(user_turn)
        self.history.append(assistant_turn)

        # Update AgentSession in ContextLake
        session = self.lake.get_agent_session(self.session_id)
        session_tokens = 0
        if session is not None:
            turn_tokens = estimate_tokens(message) + estimate_tokens(answer)
            session.used_tokens += turn_tokens
            session.last_processed_turn += 2
            session.updated_at = time.time()
            if self.persist_turns:
                session.metadata["turn_count"] = len(self.history)
                session.metadata["last_intent"] = intent
            self.lake.upsert_agent_session(session)
            session_tokens = session.used_tokens

        return {
            "answer": answer,
            "intent": intent,
            "cited_nodes": cited_nodes,
            "subgraph": subgraph,
            "turn": assistant_turn,
            "session_tokens": session_tokens,
        }

    # -------------------------------------------------------------------------
    # Rich Formatting & Interactive REPL
    # -------------------------------------------------------------------------

    def format_turn_rich(self, response: Dict[str, Any]) -> str:
        """Formats the conversational response with citations and subgraph provenance for CLI/TUI display."""
        lines = []
        intent = response.get("intent", "query")
        answer = response.get("answer", "")
        cited_nodes = response.get("cited_nodes", [])
        subgraph = response.get("subgraph", {})
        session_tokens = response.get("session_tokens", 0)

        lines.append(f"[{intent.upper()}] {answer}")

        if cited_nodes and intent not in ("list", "window_status"):
            lines.append("\nCitations:")
            for n in cited_nodes:
                nid = n.get("node_id", "")[:8]
                cname = n.get("canonical_name", "")
                cat = n.get("category", "")
                conf = n.get("confidence", 1.0)
                src = n.get("source_agent_id", "")
                lines.append(f"  • [{nid}] {cname} ({cat}, conf={conf:.2f}, src={src})")

        edges = subgraph.get("edges", [])
        if edges and intent not in ("list", "window_status"):
            lines.append("\nGraph Provenance:")
            for e in edges:
                src = e.get("source_id", "")[:8]
                tgt = e.get("target_id", "")[:8]
                rel = e.get("relation_type", "relates_to")
                lines.append(f"  • [{src}] --[{rel}]--> [{tgt}]")

        if session_tokens > 0:
            lines.append(f"\n[Session Tokens: {session_tokens:,}]")

        return "\n".join(lines)

    def run_repl(
        self,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> int:
        """
        Interactive multi-turn conversational REPL supporting /help, /list, /remember,
        /forget, /update, /as-of, /windows, /history, and /exit (quit/q) as well as natural language conversation.
        """
        output_fn("=== Context Palace Conversational Memory Palace v2.0 ===")
        output_fn("Commands: /remember <fact>, /forget <target>, /update <target> to <new>,")
        output_fn("          /list, /as-of <timestamp> <query>, /windows, /history, /exit")
        output_fn("Type any natural language query to chat with the federated memory lake.\n")

        while True:
            try:
                line = input_fn("context-palace> ")
            except (EOFError, KeyboardInterrupt):
                output_fn("\nExiting conversational REPL. Goodbye!")
                return 0

            if line is None:
                return 0
            cleaned = line.strip()
            if not cleaned:
                continue

            if cleaned.lower() in ("/exit", "/quit", "/q", "exit", "quit", "q"):
                output_fn("Exiting conversational REPL. Goodbye!")
                return 0

            if cleaned.lower() in ("/help", "help"):
                output_fn("\nAvailable Commands:")
                output_fn("  /remember <fact>                  Store a new memory")
                output_fn("  /forget <target>                  Soft-delete / invalidate a memory")
                output_fn("  /update <target> to <content>     Supersede an existing memory")
                output_fn("  /list                             List all active & superseded memories")
                output_fn("  /as-of <timestamp> <query>        Query historical state at timestamp")
                output_fn("  /windows                          Inspect model context window budget")
                output_fn("  /history                          Show conversation turn history")
                output_fn("  /exit                             Exit REPL\n")
                continue

            if cleaned.lower() in ("/history", "history"):
                output_fn(f"\nConversation History ({len(self.history)} turns):")
                for turn in self.history:
                    output_fn(f"  [{turn.role.upper()} - {turn.intent}]: {turn.message[:120]}")
                output_fn("")
                continue

            resp = self.ask(cleaned)
            formatted = self.format_turn_rich(resp)
            output_fn(formatted)
            output_fn("")

        return 0
