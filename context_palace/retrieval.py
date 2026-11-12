"""
Dual-Direction Retrieval (Proactive Pull + Reactive Push) and Sidecar Capsule Injection
for the Federated Context Lake (context-palace).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import re

from context_palace.models import (
    AgentSession,
    ConceptNode,
    ContextCapsule,
    RelationEdge,
    compute_text_embedding,
    cosine_similarity,
    estimate_tokens,
)

CAPSULE_START_MARKER = "<!-- CONTEXT_PALACE_CAPSULE_START -->"
CAPSULE_END_MARKER = "<!-- CONTEXT_PALACE_CAPSULE_END -->"


def _coerce_node(raw: Any) -> ConceptNode:
    if isinstance(raw, ConceptNode):
        return raw
    if isinstance(raw, dict):
        return ConceptNode(
            node_id=str(raw.get("node_id", raw.get("id", ""))),
            canonical_name=str(raw.get("canonical_name", "")),
            category=str(raw.get("category", "entity")),
            active_value=str(raw.get("active_value", raw.get("value", ""))),
            summary=str(raw.get("summary", "")),
            confidence=float(raw.get("confidence", 0.9)),
            pinned=bool(raw.get("pinned", False)),
            embedding=list(raw.get("embedding", [])),
            source_agent_id=str(raw.get("source_agent_id", "system")),
            created_at=float(raw.get("created_at", 0.0)),
            updated_at=float(raw.get("updated_at", 0.0)),
            tombstoned=bool(raw.get("tombstoned", False)),
        )
    raise TypeError(f"Cannot coerce {type(raw)} to ConceptNode")


def _coerce_edge(raw: Any) -> RelationEdge:
    if isinstance(raw, RelationEdge):
        return raw
    if isinstance(raw, dict):
        return RelationEdge(
            edge_id=str(raw.get("edge_id", raw.get("id", ""))),
            source_id=str(raw.get("source_id", "")),
            target_id=str(raw.get("target_id", "")),
            relation_type=str(raw.get("relation_type", "depends_on")),
            weight=float(raw.get("weight", 1.0)),
            source_agent_id=str(raw.get("source_agent_id", "system")),
            valid_from=float(raw.get("valid_from", 0.0)),
            valid_to=raw.get("valid_to"),
        )
    raise TypeError(f"Cannot coerce {type(raw)} to RelationEdge")


class DualDirectionRetriever:
    """
    Executes both:
    1. Proactive Pull (Agent -> Lake): Subgraph retrieval with vector similarity, keyword matching,
       confidence weighting, pinned priority, and multi-hop graph traversal.
    2. Reactive Push (Lake -> Agent): Model-Window Adaptive Context Capsule construction and
       sidecar file injection.
    """

    def __init__(self, lake: Any) -> None:
        self.lake = lake

    def _load_graph(self, as_of: Optional[float] = None) -> Tuple[List[ConceptNode], List[RelationEdge]]:
        if as_of is not None and hasattr(self.lake, "snapshot"):
            snap = self.lake.snapshot(as_of=as_of)
            if isinstance(snap, dict):
                raw_nodes = snap.get("nodes", snap.get("concepts", []))
                raw_edges = snap.get("edges", snap.get("relations", []))
            elif isinstance(snap, (tuple, list)) and len(snap) == 2:
                raw_nodes, raw_edges = snap[0], snap[1]
            else:
                raw_nodes, raw_edges = [], []
            nodes = [_coerce_node(n) for n in raw_nodes]
            edges = [_coerce_edge(e) for e in raw_edges]
            nodes = [n for n in nodes if not n.tombstoned]
            edges = [e for e in edges if e.valid_to is None or e.valid_to > as_of]
            return nodes, edges

        raw_nodes = self.lake.list_concepts(include_tombstoned=False) if hasattr(self.lake, "list_concepts") else []
        raw_edges = self.lake.list_edges(active_only=True) if hasattr(self.lake, "list_edges") else []
        nodes = [_coerce_node(n) for n in raw_nodes if not getattr(n, "tombstoned", False)]
        edges = [_coerce_edge(e) for e in raw_edges]
        return nodes, edges

    def score_node(self, node: ConceptNode, query: str, query_vec: List[float], include_pinned: bool = True) -> float:
        """Compute composite relevance score for a ConceptNode against a query."""
        score = 0.0

        # 1. Pinned priority
        if include_pinned and node.pinned:
            score += 1.0

        # 2. Confidence weight
        score += float(node.confidence) * 0.1

        if query and query.strip():
            # 3. Cosine similarity
            node_text = f"{node.canonical_name} {node.category} {node.active_value} {node.summary}".strip()
            node_vec = node.embedding if node.embedding else compute_text_embedding(node_text)
            sim = cosine_similarity(query_vec, node_vec)
            score += max(0.0, sim)

            # 4. Keyword / substring match bonus
            q_lower = query.lower().strip()
            q_tokens = [t for t in re.split(r"\W+", q_lower) if len(t) >= 2]
            name_lower = node.canonical_name.lower()
            val_lower = node.active_value.lower()
            cat_lower = node.category.lower()
            sum_lower = node.summary.lower()

            if q_lower in name_lower:
                score += 0.65
            if q_lower in val_lower or q_lower in sum_lower:
                score += 0.45
            if q_lower in cat_lower:
                score += 0.30

            if q_tokens:
                matched_tokens = 0
                for tok in q_tokens:
                    if tok in name_lower:
                        score += 0.25
                        matched_tokens += 1
                    elif tok in val_lower or tok in sum_lower:
                        score += 0.15
                        matched_tokens += 1
                    elif tok in cat_lower:
                        score += 0.10
                        matched_tokens += 1
                score += (matched_tokens / len(q_tokens)) * 0.25

        return round(score, 6)

    def query_subgraph(
        self,
        query: str,
        top_k: int = 8,
        hops: int = 1,
        as_of: Optional[float] = None,
        include_pinned: bool = True,
    ) -> Tuple[List[ConceptNode], List[RelationEdge]]:
        """
        Proactive Pull (Agent -> Lake):
        Scores concepts by vector similarity, keyword match, confidence, and pinned flag,
        selects top_k seeds, and expands `hops` along incident edges.
        """
        nodes, edges = self._load_graph(as_of=as_of)
        if not nodes:
            return [], []

        node_by_id: Dict[str, ConceptNode] = {n.node_id: n for n in nodes}
        query_vec = compute_text_embedding(query) if query else []

        scored: List[Tuple[float, ConceptNode]] = [
            (self.score_node(n, query, query_vec, include_pinned=include_pinned), n)
            for n in nodes
        ]
        scored.sort(key=lambda pair: (pair[0], pair[1].confidence, pair[1].updated_at), reverse=True)

        seed_count = max(1, top_k) if top_k > 0 else 0
        seed_nodes = [n for _, n in scored[:seed_count]]
        selected_ids: Set[str] = {n.node_id for n in seed_nodes}

        # Multi-hop graph expansion along incident edges
        frontier: Set[str] = set(selected_ids)
        for _ in range(max(0, hops)):
            if not frontier:
                break
            next_frontier: Set[str] = set()
            for edge in edges:
                if edge.source_id in frontier and edge.target_id in node_by_id and edge.target_id not in selected_ids:
                    selected_ids.add(edge.target_id)
                    next_frontier.add(edge.target_id)
                elif edge.target_id in frontier and edge.source_id in node_by_id and edge.source_id not in selected_ids:
                    selected_ids.add(edge.source_id)
                    next_frontier.add(edge.source_id)
            frontier = next_frontier

        # Preserve relevance ordering: seeds first in score order, then expanded neighbors by score
        score_map: Dict[str, float] = {n.node_id: sc for sc, n in scored}
        seed_id_set = {n.node_id for n in seed_nodes}
        selected_nodes = sorted(
            [node_by_id[nid] for nid in selected_ids if nid in node_by_id],
            key=lambda n: (
                1 if (include_pinned and n.pinned) else 0,
                1 if n.node_id in seed_id_set else 0,
                score_map.get(n.node_id, 0.0),
                n.confidence,
            ),
            reverse=True,
        )

        induced_edges = [
            e for e in edges if e.source_id in selected_ids and e.target_id in selected_ids
        ]

        return selected_nodes, induced_edges

    def _render_capsule_markdown(
        self,
        session: AgentSession,
        packed_nodes: List[ConceptNode],
        packed_edges: List[RelationEdge],
        node_name_by_id: Dict[str, str],
    ) -> str:
        lines = [
            CAPSULE_START_MARKER,
            f"### Federated Context Capsule (`{session.agent_id}` | Model: `{session.model_id}`)",
            (
                f"- **Model Window:** `{session.used_tokens:,} / {session.max_context_tokens:,} tokens "
                f"({session.utilization_pct:.1f}%)` | **Headroom:** `{session.remaining_headroom:,} tokens`"
            ),
            f"- **Active Facts ({len(packed_nodes)} nodes, {len(packed_edges)} edges):**",
        ]
        for n in packed_nodes:
            pin_badge = "[PINNED] " if n.pinned else ""
            summary_suffix = f" ({n.summary})" if n.summary and n.summary != n.active_value else ""
            lines.append(
                f"  - {pin_badge}**{n.canonical_name}** [`{n.category}`]: {n.active_value}{summary_suffix} "
                f"*(conf={n.confidence:.2f}, src={n.source_agent_id})*"
            )
        if packed_edges:
            lines.append("- **Relations:**")
            for e in packed_edges:
                src_name = node_name_by_id.get(e.source_id, e.source_id)
                tgt_name = node_name_by_id.get(e.target_id, e.target_id)
                lines.append(f"  - `{src_name}` --[{e.relation_type}]--> `{tgt_name}`")
        lines.append(CAPSULE_END_MARKER)
        return "\n".join(lines)

    def build_context_capsule(
        self,
        session: AgentSession,
        task_query: str = "",
        budget_tokens: Optional[int] = None,
        raw_history_tokens: Optional[int] = None,
        as_of: Optional[float] = None,
    ) -> ContextCapsule:
        """
        Reactive Push (Lake -> Agent) with Model-Window Adaptive Budgeting.
        Packs pinned and highest-relevance concepts + induced edges within `allocated_budget` tokens.
        """
        allocated_budget = (
            budget_tokens if budget_tokens is not None else session.adaptive_capsule_budget()
        )
        candidate_nodes, candidate_edges = self.query_subgraph(
            query=task_query,
            top_k=20,
            hops=1,
            as_of=as_of,
            include_pinned=True,
        )

        query_vec = compute_text_embedding(task_query) if task_query else []
        ordered_candidates = sorted(
            candidate_nodes,
            key=lambda n: (
                1 if n.pinned else 0,
                self.score_node(n, task_query, query_vec, include_pinned=True),
                n.confidence,
            ),
            reverse=True,
        )

        node_name_by_id: Dict[str, str] = {n.node_id: n.canonical_name for n in ordered_candidates}
        packed_nodes: List[ConceptNode] = []
        packed_edges: List[RelationEdge] = []
        packed_ids: Set[str] = set()

        for node in ordered_candidates:
            trial_nodes = packed_nodes + [node]
            trial_ids = packed_ids | {node.node_id}
            trial_edges = [
                e for e in candidate_edges if e.source_id in trial_ids and e.target_id in trial_ids
            ]
            trial_md = self._render_capsule_markdown(session, trial_nodes, trial_edges, node_name_by_id)
            trial_tokens = estimate_tokens(trial_md)

            if trial_tokens <= allocated_budget or not packed_nodes:
                # Always allow at least the first highest-priority node if budget > 0 and it fits or is minimal
                if trial_tokens > allocated_budget and packed_nodes:
                    break
                if trial_tokens <= allocated_budget:
                    packed_nodes = trial_nodes
                    packed_ids = trial_ids
                    packed_edges = trial_edges
                else:
                    # Try adding the node without edges if edges pushed it over budget
                    trial_md_no_edges = self._render_capsule_markdown(
                        session, trial_nodes, packed_edges, node_name_by_id
                    )
                    if estimate_tokens(trial_md_no_edges) <= allocated_budget:
                        packed_nodes = trial_nodes
                        packed_ids = trial_ids
                    elif not packed_nodes:
                        # Truncate active_value to fit tight budget so capsule is still valid
                        truncated_node = ConceptNode(
                            node_id=node.node_id,
                            canonical_name=node.canonical_name,
                            category=node.category,
                            active_value=node.active_value[:64],
                            summary="",
                            confidence=node.confidence,
                            pinned=node.pinned,
                            embedding=node.embedding,
                            source_agent_id=node.source_agent_id,
                            created_at=node.created_at,
                            updated_at=node.updated_at,
                            tombstoned=node.tombstoned,
                        )
                        packed_nodes = [truncated_node]
                        packed_ids = {truncated_node.node_id}
                        break
                    else:
                        break

        markdown_payload = self._render_capsule_markdown(
            session, packed_nodes, packed_edges, node_name_by_id
        )
        capsule_tokens = estimate_tokens(markdown_payload)
        raw_equiv = (
            raw_history_tokens
            if raw_history_tokens is not None
            else max(session.used_tokens, capsule_tokens * 12, 2000)
        )
        token_reduction_pct = round(
            max(0.0, (1.0 - (capsule_tokens / max(1, raw_equiv))) * 100.0), 2
        )

        return ContextCapsule(
            target_agent_id=session.agent_id,
            target_model_id=session.model_id,
            max_context_tokens=session.max_context_tokens,
            remaining_headroom=session.remaining_headroom,
            allocated_budget_tokens=allocated_budget,
            capsule_tokens=capsule_tokens,
            raw_history_tokens_equivalent=raw_equiv,
            token_reduction_pct=token_reduction_pct,
            nodes=packed_nodes,
            edges=packed_edges,
            markdown_payload=markdown_payload,
            as_of_timestamp=as_of,
        )

    def inject_into_sidecar(self, capsule: ContextCapsule, target_path: str) -> str:
        """
        Idempotently creates or updates `target_path` (e.g. `.context_capsule.md` or `AGENTS.md`).
        If `<!-- CONTEXT_PALACE_CAPSULE_START -->` ... `<!-- CONTEXT_PALACE_CAPSULE_END -->`
        already exists in the file, replaces that block in-place; otherwise appends it cleanly.
        """
        path = Path(target_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = capsule.markdown_payload.strip()

        if path.exists():
            existing = path.read_text(encoding="utf-8")
            start_idx = existing.find(CAPSULE_START_MARKER)
            end_idx = existing.find(CAPSULE_END_MARKER)
            if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                end_pos = end_idx + len(CAPSULE_END_MARKER)
                updated = existing[:start_idx] + payload + existing[end_pos:]
            else:
                sep = "\n\n" if existing and not existing.endswith("\n\n") else ""
                if existing and existing.endswith("\n") and not existing.endswith("\n\n"):
                    sep = "\n"
                updated = f"{existing}{sep}{payload}\n"
        else:
            updated = f"{payload}\n"

        path.write_text(updated, encoding="utf-8")
        return updated
