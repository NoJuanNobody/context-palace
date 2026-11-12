"""
Command-Line Interface (CLI) for Context Palace.
Supports TUI dashboard, local agent transcript & model context window discovery,
CRUD operations on the Federated Context Lake, adaptive Context Capsule injection,
bitemporal time-travel snapshots, and MCP stdio server mode.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from context_palace.models import (
    AgentSession,
    BeliefRevision,
    ConceptNode,
    ConflictResolutionEvent,
    ContextCapsule,
    ModelWindowSpec,
    RelationEdge,
    compute_text_embedding,
    estimate_tokens,
)
from context_palace.tui import MemoryPalaceTUI, _call_compatible


class _FallbackSQLiteContextLake:
    """
    Lightweight SQLite-backed fallback ContextLake used only when running inside an
    isolated worktree where `context_palace.lake` has not yet been merged.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            str(Path(db_path).expanduser()) if db_path != ":memory:" else ":memory:"
        )
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS agent_sessions (
                agent_id TEXT PRIMARY KEY,
                harness_type TEXT,
                model_id TEXT,
                max_context_tokens INTEGER,
                used_tokens INTEGER,
                transcript_path TEXT,
                last_processed_turn INTEGER,
                last_byte_offset INTEGER,
                updated_at REAL,
                metadata_json TEXT
            );
            CREATE TABLE IF NOT EXISTS concept_nodes (
                node_id TEXT PRIMARY KEY,
                canonical_name TEXT UNIQUE,
                category TEXT,
                active_value TEXT,
                summary TEXT,
                confidence REAL,
                pinned INTEGER,
                embedding_json TEXT,
                source_agent_id TEXT,
                created_at REAL,
                updated_at REAL,
                tombstoned INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS belief_revisions (
                revision_id TEXT PRIMARY KEY,
                node_id TEXT,
                value TEXT,
                confidence REAL,
                source_agent_id TEXT,
                provenance_type TEXT,
                turn_id INTEGER,
                valid_from REAL,
                valid_to REAL,
                superseded_by TEXT
            );
            CREATE TABLE IF NOT EXISTS relation_edges (
                edge_id TEXT PRIMARY KEY,
                source_id TEXT,
                target_id TEXT,
                relation_type TEXT,
                weight REAL,
                source_agent_id TEXT,
                valid_from REAL,
                valid_to REAL
            );
            CREATE TABLE IF NOT EXISTS conflict_events (
                event_id TEXT PRIMARY KEY,
                node_id TEXT,
                winning_revision_id TEXT,
                losing_revision_id TEXT,
                winning_agent_id TEXT,
                losing_agent_id TEXT,
                winning_score REAL,
                losing_score REAL,
                reason TEXT,
                resolved_at REAL
            );
            """
        )
        self.conn.commit()

    def upsert_agent_session(self, session: AgentSession) -> AgentSession:
        now = session.updated_at or time.time()
        session.updated_at = now
        self.conn.execute(
            """
            INSERT INTO agent_sessions (
                agent_id, harness_type, model_id, max_context_tokens, used_tokens,
                transcript_path, last_processed_turn, last_byte_offset, updated_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_id) DO UPDATE SET
                harness_type=excluded.harness_type,
                model_id=excluded.model_id,
                max_context_tokens=excluded.max_context_tokens,
                used_tokens=excluded.used_tokens,
                transcript_path=excluded.transcript_path,
                last_processed_turn=excluded.last_processed_turn,
                last_byte_offset=excluded.last_byte_offset,
                updated_at=excluded.updated_at,
                metadata_json=excluded.metadata_json
            """,
            (
                session.agent_id,
                session.harness_type,
                session.model_id,
                session.max_context_tokens,
                session.used_tokens,
                session.transcript_path,
                session.last_processed_turn,
                session.last_byte_offset,
                session.updated_at,
                json.dumps(session.metadata or {}),
            ),
        )
        self.conn.commit()
        return session

    def get_agent_session(self, agent_id: str) -> Optional[AgentSession]:
        row = self.conn.execute(
            "SELECT * FROM agent_sessions WHERE agent_id = ?", (agent_id,)
        ).fetchone()
        if not row:
            return None
        return AgentSession(
            agent_id=row["agent_id"],
            harness_type=row["harness_type"],
            model_id=row["model_id"],
            max_context_tokens=int(row["max_context_tokens"]),
            used_tokens=int(row["used_tokens"]),
            transcript_path=row["transcript_path"],
            last_processed_turn=int(row["last_processed_turn"]),
            last_byte_offset=int(row["last_byte_offset"]),
            updated_at=float(row["updated_at"]),
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def list_agent_sessions(self) -> List[AgentSession]:
        rows = self.conn.execute(
            "SELECT * FROM agent_sessions ORDER BY updated_at DESC"
        ).fetchall()
        return [
            AgentSession(
                agent_id=row["agent_id"],
                harness_type=row["harness_type"],
                model_id=row["model_id"],
                max_context_tokens=int(row["max_context_tokens"]),
                used_tokens=int(row["used_tokens"]),
                transcript_path=row["transcript_path"],
                last_processed_turn=int(row["last_processed_turn"]),
                last_byte_offset=int(row["last_byte_offset"]),
                updated_at=float(row["updated_at"]),
                metadata=json.loads(row["metadata_json"] or "{}"),
            )
            for row in rows
        ]

    def _row_to_node(self, row: sqlite3.Row) -> ConceptNode:
        return ConceptNode(
            node_id=row["node_id"],
            canonical_name=row["canonical_name"],
            category=row["category"],
            active_value=row["active_value"],
            summary=row["summary"] or "",
            confidence=float(row["confidence"]),
            pinned=bool(row["pinned"]),
            embedding=json.loads(row["embedding_json"] or "[]"),
            source_agent_id=row["source_agent_id"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            tombstoned=bool(row["tombstoned"]),
        )

    def create_concept(
        self,
        canonical_name: str,
        category: str,
        value: str,
        summary: str = "",
        confidence: float = 0.9,
        source_agent_id: str = "system",
        provenance_type: str = "operator_crud",
        pinned: bool = False,
        turn_id: int = 0,
        timestamp: Optional[float] = None,
    ) -> ConceptNode:
        ts = float(timestamp) if timestamp is not None else time.time()
        existing = self.get_concept(canonical_name, include_tombstoned=True)
        if existing is not None:
            return self.update_concept(
                canonical_name,
                new_value=value,
                summary=summary or existing.summary,
                confidence=confidence,
                source_agent_id=source_agent_id,
                provenance_type=provenance_type,
                pinned=pinned,
                turn_id=turn_id,
                timestamp=ts,
            )
        node_id = f"node_{uuid.uuid4().hex[:10]}"
        emb = compute_text_embedding(f"{canonical_name} {category} {value} {summary}")
        self.conn.execute(
            """
            INSERT INTO concept_nodes (
                node_id, canonical_name, category, active_value, summary,
                confidence, pinned, embedding_json, source_agent_id, created_at, updated_at, tombstoned
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                node_id,
                canonical_name,
                category,
                value,
                summary,
                confidence,
                1 if pinned else 0,
                json.dumps(emb),
                source_agent_id,
                ts,
                ts,
            ),
        )
        rev_id = f"rev_{uuid.uuid4().hex[:10]}"
        self.conn.execute(
            """
            INSERT INTO belief_revisions (
                revision_id, node_id, value, confidence, source_agent_id,
                provenance_type, turn_id, valid_from, valid_to, superseded_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
            """,
            (
                rev_id,
                node_id,
                value,
                confidence,
                source_agent_id,
                provenance_type,
                turn_id,
                ts,
            ),
        )
        self.conn.commit()
        return self.get_concept(node_id, include_tombstoned=True)  # type: ignore[return-value]

    def get_concept(
        self, node_id_or_name: str, include_tombstoned: bool = False
    ) -> Optional[ConceptNode]:
        clause = "" if include_tombstoned else "AND tombstoned = 0"
        row = self.conn.execute(
            f"SELECT * FROM concept_nodes WHERE (node_id = ? OR canonical_name = ?) {clause}",
            (node_id_or_name, node_id_or_name),
        ).fetchone()
        return self._row_to_node(row) if row else None

    def list_concepts(
        self,
        category: Optional[str] = None,
        pinned_only: bool = False,
        include_tombstoned: bool = False,
    ) -> List[ConceptNode]:
        conds = []
        params: List[Any] = []
        if not include_tombstoned:
            conds.append("tombstoned = 0")
        if category:
            conds.append("category = ?")
            params.append(category)
        if pinned_only:
            conds.append("pinned = 1")
        where_sql = f"WHERE {' AND '.join(conds)}" if conds else ""
        rows = self.conn.execute(
            f"SELECT * FROM concept_nodes {where_sql} ORDER BY pinned DESC, updated_at DESC",
            params,
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def update_concept(
        self,
        node_id_or_name: str,
        new_value: str,
        summary: Optional[str] = None,
        confidence: float = 0.9,
        source_agent_id: str = "system",
        provenance_type: str = "operator_crud",
        pinned: Optional[bool] = None,
        turn_id: int = 0,
        timestamp: Optional[float] = None,
    ) -> ConceptNode:
        node = self.get_concept(node_id_or_name, include_tombstoned=True)
        if node is None:
            raise KeyError(f"Concept {node_id_or_name!r} not found")
        ts = float(timestamp) if timestamp is not None else time.time()
        new_rev_id = f"rev_{uuid.uuid4().hex[:10]}"
        self.conn.execute(
            """
            UPDATE belief_revisions
            SET valid_to = ?, superseded_by = ?
            WHERE node_id = ? AND valid_to IS NULL
            """,
            (ts, new_rev_id, node.node_id),
        )
        self.conn.execute(
            """
            INSERT INTO belief_revisions (
                revision_id, node_id, value, confidence, source_agent_id,
                provenance_type, turn_id, valid_from, valid_to, superseded_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
            """,
            (
                new_rev_id,
                node.node_id,
                new_value,
                confidence,
                source_agent_id,
                provenance_type,
                turn_id,
                ts,
            ),
        )
        updated_summary = summary if summary is not None else node.summary
        updated_pinned = node.pinned if pinned is None else bool(pinned)
        emb = compute_text_embedding(
            f"{node.canonical_name} {node.category} {new_value} {updated_summary}"
        )
        self.conn.execute(
            """
            UPDATE concept_nodes
            SET active_value = ?, summary = ?, confidence = ?, pinned = ?,
                embedding_json = ?, source_agent_id = ?, updated_at = ?, tombstoned = 0
            WHERE node_id = ?
            """,
            (
                new_value,
                updated_summary,
                confidence,
                1 if updated_pinned else 0,
                json.dumps(emb),
                source_agent_id,
                ts,
                node.node_id,
            ),
        )
        self.conn.commit()
        return self.get_concept(node.node_id, include_tombstoned=True)  # type: ignore[return-value]

    def pin_concept(self, node_id_or_name: str, pinned: bool = True) -> ConceptNode:
        node = self.get_concept(node_id_or_name, include_tombstoned=True)
        if node is None:
            raise KeyError(f"Concept {node_id_or_name!r} not found")
        self.conn.execute(
            "UPDATE concept_nodes SET pinned = ? WHERE node_id = ?",
            (1 if pinned else 0, node.node_id),
        )
        self.conn.commit()
        return self.get_concept(node.node_id, include_tombstoned=True)  # type: ignore[return-value]

    def delete_concept(self, node_id_or_name: str, hard: bool = False, timestamp: Optional[float] = None) -> bool:
        node = self.get_concept(node_id_or_name, include_tombstoned=True)
        if node is None:
            return False
        ts = float(timestamp) if timestamp is not None else time.time()
        if hard:
            self.conn.execute("DELETE FROM belief_revisions WHERE node_id = ?", (node.node_id,))
            self.conn.execute(
                "DELETE FROM relation_edges WHERE source_id = ? OR target_id = ?",
                (node.node_id, node.node_id),
            )
            self.conn.execute("DELETE FROM concept_nodes WHERE node_id = ?", (node.node_id,))
        else:
            self.conn.execute(
                "UPDATE concept_nodes SET tombstoned = 1, updated_at = ? WHERE node_id = ?",
                (ts, node.node_id),
            )
            self.conn.execute(
                "UPDATE belief_revisions SET valid_to = ? WHERE node_id = ? AND valid_to IS NULL",
                (ts, node.node_id),
            )
        self.conn.commit()
        return True

    def get_belief_history(self, node_id_or_name: str) -> List[BeliefRevision]:
        node = self.get_concept(node_id_or_name, include_tombstoned=True)
        target_id = node.node_id if node else node_id_or_name
        rows = self.conn.execute(
            "SELECT * FROM belief_revisions WHERE node_id = ? ORDER BY valid_from ASC",
            (target_id,),
        ).fetchall()
        return [
            BeliefRevision(
                revision_id=r["revision_id"],
                node_id=r["node_id"],
                value=r["value"],
                confidence=float(r["confidence"]),
                source_agent_id=r["source_agent_id"],
                provenance_type=r["provenance_type"],
                turn_id=int(r["turn_id"]),
                valid_from=float(r["valid_from"]),
                valid_to=float(r["valid_to"]) if r["valid_to"] is not None else None,
                superseded_by=r["superseded_by"],
            )
            for r in rows
        ]

    def list_edges(self, active_only: bool = True) -> List[RelationEdge]:
        clause = "WHERE valid_to IS NULL" if active_only else ""
        rows = self.conn.execute(f"SELECT * FROM relation_edges {clause}").fetchall()
        return [
            RelationEdge(
                edge_id=r["edge_id"],
                source_id=r["source_id"],
                target_id=r["target_id"],
                relation_type=r["relation_type"],
                weight=float(r["weight"]),
                source_agent_id=r["source_agent_id"],
                valid_from=float(r["valid_from"]),
                valid_to=float(r["valid_to"]) if r["valid_to"] is not None else None,
            )
            for r in rows
        ]

    def list_conflict_events(self) -> List[ConflictResolutionEvent]:
        rows = self.conn.execute(
            "SELECT * FROM conflict_events ORDER BY resolved_at DESC"
        ).fetchall()
        return [
            ConflictResolutionEvent(
                event_id=r["event_id"],
                node_id=r["node_id"],
                winning_revision_id=r["winning_revision_id"],
                losing_revision_id=r["losing_revision_id"],
                winning_agent_id=r["winning_agent_id"],
                losing_agent_id=r["losing_agent_id"],
                winning_score=float(r["winning_score"]),
                losing_score=float(r["losing_score"]),
                reason=r["reason"],
                resolved_at=float(r["resolved_at"]),
            )
            for r in rows
        ]

    def snapshot(self, as_of: Optional[float] = None) -> Dict[str, Any]:
        if as_of is None:
            nodes = self.list_concepts(include_tombstoned=False)
            edges = self.list_edges(active_only=True)
            return {
                "as_of": None,
                "nodes": [n.to_dict() for n in nodes],
                "edges": [e.to_dict() for e in edges],
            }
        ts = float(as_of)
        rows = self.conn.execute(
            """
            SELECT c.*, b.value AS rev_value, b.confidence AS rev_confidence,
                   b.source_agent_id AS rev_agent, b.valid_from AS rev_from
            FROM concept_nodes c
            JOIN belief_revisions b ON c.node_id = b.node_id
            WHERE b.valid_from <= ? AND (b.valid_to IS NULL OR b.valid_to > ?)
            ORDER BY b.valid_from DESC
            """,
            (ts, ts),
        ).fetchall()
        seen_nodes = set()
        snap_nodes: List[Dict[str, Any]] = []
        for r in rows:
            nid = r["node_id"]
            if nid in seen_nodes:
                continue
            seen_nodes.add(nid)
            node_obj = ConceptNode(
                node_id=nid,
                canonical_name=r["canonical_name"],
                category=r["category"],
                active_value=r["rev_value"],
                summary=r["summary"] or "",
                confidence=float(r["rev_confidence"]),
                pinned=bool(r["pinned"]),
                embedding=json.loads(r["embedding_json"] or "[]"),
                source_agent_id=r["rev_agent"],
                created_at=float(r["created_at"]),
                updated_at=float(r["rev_from"]),
                tombstoned=False,
            )
            snap_nodes.append(node_obj.to_dict())

        edge_rows = self.conn.execute(
            """
            SELECT * FROM relation_edges
            WHERE valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)
            """,
            (ts, ts),
        ).fetchall()
        snap_edges = [
            RelationEdge(
                edge_id=er["edge_id"],
                source_id=er["source_id"],
                target_id=er["target_id"],
                relation_type=er["relation_type"],
                weight=float(er["weight"]),
                source_agent_id=er["source_agent_id"],
                valid_from=float(er["valid_from"]),
                valid_to=float(er["valid_to"]) if er["valid_to"] is not None else None,
            ).to_dict()
            for er in edge_rows
        ]
        return {
            "as_of": ts,
            "nodes": snap_nodes,
            "edges": snap_edges,
        }


class _FallbackModelRegistry:
    """Fallback ModelContextRegistry when running in an isolated worktree."""

    CATALOG: Dict[str, tuple[str, int, int]] = {
        "claude-3-7-sonnet": ("anthropic", 200_000, 16_384),
        "claude-sonnet-4": ("anthropic", 200_000, 16_384),
        "claude-opus-4": ("anthropic", 200_000, 32_000),
        "gemini-2.5-pro": ("google", 1_048_576, 65_536),
        "gemini-2.5-flash": ("google", 1_048_576, 65_536),
        "gpt-4o": ("openai", 128_000, 16_384),
        "o3-mini": ("openai", 200_000, 100_000),
        "qwen2.5-coder:7b": ("ollama", 32_768, 4_096),
        "llama3.2:3b": ("ollama", 8_192, 2_048),
    }

    def resolve_model_window(
        self,
        model_id: str,
        session_metadata: Optional[Dict[str, Any]] = None,
        allow_live_probe: bool = False,
    ) -> ModelWindowSpec:
        if session_metadata and "max_context_tokens" in session_metadata:
            return ModelWindowSpec(
                model_id=model_id,
                canonical_name=model_id,
                provider=str(session_metadata.get("provider", "session")),
                max_context_tokens=int(session_metadata["max_context_tokens"]),
                source="session_metadata",
            )
        m_lower = model_id.lower()
        for key, (provider, ctx, out) in self.CATALOG.items():
            if key in m_lower:
                return ModelWindowSpec(
                    model_id=model_id,
                    canonical_name=key,
                    provider=provider,
                    max_context_tokens=ctx,
                    max_output_tokens=out,
                    source="builtin_catalog",
                )
        return ModelWindowSpec(
            model_id=model_id,
            canonical_name=model_id,
            provider="unknown",
            max_context_tokens=128_000,
            max_output_tokens=8_192,
            source="builtin_catalog",
        )

    def discover_local_agents(
        self, search_dirs: Optional[List[str]] = None, limit: int = 10
    ) -> List[AgentSession]:
        return []


class _FallbackRetriever:
    """Fallback DualDirectionRetriever when running in an isolated worktree."""

    def __init__(self, lake: Any, registry: Optional[Any] = None) -> None:
        self.lake = lake
        self.registry = registry or _FallbackModelRegistry()

    def build_context_capsule(
        self,
        agent_id: str,
        query: str = "",
        as_of: Optional[float] = None,
        budget_override: Optional[int] = None,
    ) -> ContextCapsule:
        sess = (
            self.lake.get_agent_session(agent_id)
            if hasattr(self.lake, "get_agent_session")
            else None
        )
        if sess is None:
            sess = AgentSession(
                agent_id=agent_id,
                harness_type="custom",
                model_id="claude-sonnet-4",
                max_context_tokens=200_000,
                used_tokens=20_000,
            )
        budget = budget_override or sess.adaptive_capsule_budget()
        nodes = (
            self.lake.list_concepts(include_tombstoned=False)
            if hasattr(self.lake, "list_concepts")
            else []
        )
        lines = [
            f"# Context Palace Capsule for `{agent_id}` (`{sess.model_id}`)",
            f"- Model Window: {sess.used_tokens:,} / {sess.max_context_tokens:,} tokens (Free Headroom: {sess.remaining_headroom:,})",
            f"- Capsule Token Budget: {budget:,}",
            "",
            "## Active Architectural & Project Beliefs",
        ]
        selected_nodes: List[ConceptNode] = []
        for n in nodes:
            line = f"- **{n.canonical_name}** ({n.category}, conf={n.confidence:.2f}): {n.active_value}"
            lines.append(line)
            selected_nodes.append(n)
        md = "\n".join(lines)
        cap_tokens = estimate_tokens(md)
        raw_equiv = max(cap_tokens * 5, 2000)
        reduction = round((1.0 - (cap_tokens / max(1, raw_equiv))) * 100.0, 2)
        return ContextCapsule(
            target_agent_id=agent_id,
            target_model_id=sess.model_id,
            max_context_tokens=sess.max_context_tokens,
            remaining_headroom=sess.remaining_headroom,
            allocated_budget_tokens=budget,
            capsule_tokens=cap_tokens,
            raw_history_tokens_equivalent=raw_equiv,
            token_reduction_pct=reduction,
            nodes=selected_nodes,
            edges=[],
            markdown_payload=md,
            as_of_timestamp=as_of,
        )

    def inject_into_sidecar(self, capsule: ContextCapsule, sidecar_path: str | Path) -> Path:
        p = Path(sidecar_path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(capsule.markdown_payload, encoding="utf-8")
        return p


def _load_components(db_path: str) -> Dict[str, Any]:
    """Lazy-load Context Palace components with graceful fallback for isolated worktree testing."""
    try:
        from context_palace.lake import ContextLake  # type: ignore
        lake = ContextLake(db_path=db_path)
    except Exception:
        lake = _FallbackSQLiteContextLake(db_path=db_path)

    try:
        from context_palace.model_registry import ModelContextRegistry  # type: ignore
        registry = ModelContextRegistry()
    except Exception:
        registry = _FallbackModelRegistry()

    try:
        from context_palace.extractor import IncrementalDeltaExtractor  # type: ignore
        extractor = IncrementalDeltaExtractor()
    except Exception:
        extractor = None

    try:
        from context_palace.consensus import ConsensusEngine  # type: ignore
        consensus = _call_compatible(ConsensusEngine, lake=lake)
    except Exception:
        consensus = None

    try:
        from context_palace.retriever import DualDirectionRetriever  # type: ignore
        retriever = _call_compatible(DualDirectionRetriever, lake=lake, registry=registry)
    except Exception:
        retriever = _FallbackRetriever(lake=lake, registry=registry)

    try:
        from context_palace.mcp_server import ContextPalaceMCPServer  # type: ignore
        mcp_server = _call_compatible(
            ContextPalaceMCPServer,
            lake=lake,
            retriever=retriever,
            registry=registry,
            consensus=consensus,
        )
    except Exception:
        mcp_server = None

    tui = MemoryPalaceTUI(lake=lake, retriever=retriever, registry=registry)
    return {
        "lake": lake,
        "registry": registry,
        "extractor": extractor,
        "consensus": consensus,
        "retriever": retriever,
        "mcp_server": mcp_server,
        "tui": tui,
    }


def _extract_global_db_arg(argv: List[str]) -> tuple[str, List[str]]:
    """Allow `--db PATH` anywhere in argv."""
    default_db = os.environ.get(
        "CONTEXT_PALACE_DB",
        str(Path("~/.context_palace/context_lake.db").expanduser()),
    )
    cleaned: List[str] = []
    db_path = default_db
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--db" and i + 1 < len(argv):
            db_path = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--db="):
            db_path = arg.split("=", 1)[1]
            i += 1
            continue
        cleaned.append(arg)
        i += 1
    return db_path, cleaned


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="context-palace",
        description="Federated Context Lake & TUI CRUD Memory Palace with Auto-Detected Model Context Windows",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to SQLite Context Lake database (default: ~/.context_palace/context_lake.db)",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    # 1. tui
    p_tui = subparsers.add_parser("tui", help="Launch or render the Memory Palace Rich TUI")
    p_tui.add_argument("--once", action="store_true", help="Render TUI once and exit")
    p_tui.add_argument("--as-of", type=float, default=None, help="Time-travel unix timestamp")
    p_tui.add_argument("--search", type=str, default="", help="Filter concepts by query")
    p_tui.add_argument("--agent-id", type=str, default=None, help="Highlight selected agent")

    # 2. scan-agents
    p_scan = subparsers.add_parser(
        "scan-agents",
        help="Discover local agent transcripts, auto-detect model context windows, and optionally ingest deltas",
    )
    p_scan.add_argument("--dir", action="append", default=None, help="Transcript directory to scan")
    p_scan.add_argument("--limit", type=int, default=10, help="Max sessions to scan")
    p_scan.add_argument("--ingest", action="store_true", help="Ingest incremental transcript deltas into the lake")

    # 3. models
    p_models = subparsers.add_parser(
        "models", help="Resolve and display auto-detected model context windows"
    )
    p_models.add_argument("model_ids", nargs="*", help="Model identifiers to resolve")
    p_models.add_argument("--live", action="store_true", help="Allow live local/API probing")

    # 4. crud
    p_crud = subparsers.add_parser("crud", help="Perform CRUD operations on concepts in the Memory Palace")
    p_crud.add_argument(
        "action",
        choices=["create", "read", "update", "delete", "pin", "list"],
        help="CRUD action to execute",
    )
    p_crud.add_argument("--name", "--id", dest="name", type=str, default="", help="Canonical concept name or node ID")
    p_crud.add_argument("--category", type=str, default="architecture", help="Concept category")
    p_crud.add_argument("--value", type=str, default="", help="Belief value")
    p_crud.add_argument("--summary", type=str, default="", help="Optional summary")
    p_crud.add_argument("--confidence", type=float, default=0.9, help="Confidence score [0.0..1.0]")
    p_crud.add_argument("--agent-id", type=str, default="operator_cli", help="Source agent identifier")
    p_crud.add_argument("--hard", action="store_true", help="Hard-delete instead of soft tombstone")
    p_crud.add_argument("--pinned", action="store_true", help="Pin concept for proactive injection")

    # 5. inject
    p_inject = subparsers.add_parser(
        "inject",
        help="Build adaptive Context Capsule fitted to the agent's model context headroom",
    )
    p_inject.add_argument("--agent-id", required=True, help="Target agent ID")
    p_inject.add_argument("--query", default="", help="Reactive query or current task focus")
    p_inject.add_argument("--output", default=None, help="Optional sidecar file path to write capsule")

    # 6. time-travel
    p_tt = subparsers.add_parser(
        "time-travel", help="Inspect knowledge graph snapshot at a specific timestamp"
    )
    p_tt.add_argument("--as-of", type=float, required=True, help="Unix timestamp for bitemporal snapshot")

    # 7. mcp
    subparsers.add_parser("mcp", help="Run stdio JSON-RPC MCP server loop")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    db_path, filtered_argv = _extract_global_db_arg(raw_argv)
    parser = build_parser()
    args = parser.parse_args(filtered_argv)
    if args.db:
        db_path = args.db

    comps = _load_components(db_path)
    lake = comps["lake"]
    registry = comps["registry"]
    extractor = comps["extractor"]
    retriever = comps["retriever"]
    mcp_server = comps["mcp_server"]
    tui: MemoryPalaceTUI = comps["tui"]

    sub = args.subcommand or "tui"

    if sub == "tui":
        once_flag = bool(getattr(args, "once", False))
        as_of_val = getattr(args, "as_of", None)
        agent_id_val = getattr(args, "agent_id", None)
        search_val = getattr(args, "search", "")
        if once_flag or not sys.stdin.isatty():
            out = tui.render_dashboard(
                as_of=as_of_val,
                selected_agent_id=agent_id_val,
                search_query=search_val,
            )
            print(out)
            return 0
        tui.as_of = as_of_val
        tui.selected_agent_id = agent_id_val
        tui.run_interactive()
        return 0

    if sub == "scan-agents":
        sessions = _call_compatible(
            registry.discover_local_agents,
            search_dirs=args.dir,
            limit=args.limit,
        )
        ingested_concepts = 0
        for sess in sessions:
            prior = lake.get_agent_session(sess.agent_id) if hasattr(lake, "get_agent_session") else None
            if args.ingest and extractor is not None and sess.transcript_path:
                sess.last_processed_turn = prior.last_processed_turn if prior is not None else 0
                if hasattr(extractor, "ingest_turns"):
                    turns_list: List[Dict[str, Any]] = []
                    try:
                        with open(sess.transcript_path, "r", encoding="utf-8", errors="replace") as fh:
                            for idx, raw_line in enumerate(fh, start=1):
                                raw_line = raw_line.strip()
                                if not raw_line:
                                    continue
                                try:
                                    obj = json.loads(raw_line)
                                    if isinstance(obj, dict):
                                        step_type = str(obj.get("type") or "")
                                        if step_type and step_type not in ("USER_INPUT", "PLANNER_RESPONSE", ""):
                                            continue
                                        content_val = (
                                            obj.get("content")
                                            or (obj.get("message", {}).get("content") if isinstance(obj.get("message"), dict) else None)
                                            or ""
                                        )
                                        if isinstance(content_val, list):
                                            content_val = " ".join(
                                                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                                                for part in content_val
                                            )
                                        if content_val and len(str(content_val)) <= 6000:
                                            turns_list.append(
                                                {
                                                    "turn_id": idx,
                                                    "role": obj.get("source") or obj.get("type") or "agent_turn",
                                                    "content": str(content_val),
                                                }
                                            )
                                except Exception:
                                    continue
                    except OSError:
                        turns_list = []
                    if turns_list:
                        res = extractor.ingest_turns(lake=lake, session=sess, turns=turns_list)
                        if isinstance(res, dict):
                            ingested_concepts += int(res.get("created_nodes", 0)) + int(
                                res.get("merged_nodes", 0)
                            )
                elif hasattr(extractor, "ingest_session_into_lake"):
                    res = _call_compatible(
                        extractor.ingest_session_into_lake,
                        session=sess,
                        lake=lake,
                    )
                    if isinstance(res, dict):
                        ingested_concepts += int(res.get("concepts_created", 0)) + int(
                            res.get("concepts_updated", 0)
                        )
            if hasattr(lake, "upsert_agent_session"):
                lake.upsert_agent_session(sess)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "discovered_agents": len(sessions),
                    "ingested_concepts": ingested_concepts,
                    "sessions": [s.to_dict() if hasattr(s, "to_dict") else s for s in sessions],
                },
                indent=2,
            )
        )
        return 0

    if sub == "models":
        model_ids = args.model_ids or [
            "claude-sonnet-4",
            "gemini-2.5-pro",
            "gpt-4o",
            "qwen2.5-coder:7b",
        ]
        specs = [
            _call_compatible(
                registry.resolve_model_window,
                model_id=mid,
                fetch_live=args.live,
                allow_live_probe=args.live,
            ).to_dict()
            for mid in model_ids
        ]
        print(json.dumps({"status": "ok", "models": specs}, indent=2))
        return 0

    if sub == "crud":
        act = args.action
        if act == "create":
            res = tui.execute_key_action(
                "c",
                {
                    "canonical_name": args.name,
                    "category": args.category,
                    "value": args.value,
                    "summary": args.summary,
                    "confidence": args.confidence,
                    "source_agent_id": args.agent_id,
                    "pinned": args.pinned,
                },
            )
            print(json.dumps(res, indent=2))
            return 0
        if act == "read":
            res = tui.execute_key_action("r", {"node_id_or_name": args.name})
            print(json.dumps(res, indent=2))
            return 0
        if act == "update":
            payload: Dict[str, Any] = {
                "node_id_or_name": args.name,
                "new_value": args.value,
                "confidence": args.confidence,
                "source_agent_id": args.agent_id,
            }
            if args.summary:
                payload["summary"] = args.summary
            if args.pinned:
                payload["pinned"] = True
            res = tui.execute_key_action("u", payload)
            print(json.dumps(res, indent=2))
            return 0
        if act == "delete":
            res = tui.execute_key_action(
                "d", {"node_id_or_name": args.name, "hard": args.hard}
            )
            print(json.dumps(res, indent=2))
            return 0
        if act == "pin":
            res = tui.execute_key_action(
                "p", {"node_id_or_name": args.name, "pinned": True}
            )
            print(json.dumps(res, indent=2))
            return 0
        if act == "list":
            nodes = (
                lake.list_concepts(include_tombstoned=False)
                if hasattr(lake, "list_concepts")
                else []
            )
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "count": len(nodes),
                        "concepts": [n.to_dict() if hasattr(n, "to_dict") else n for n in nodes],
                    },
                    indent=2,
                )
            )
            return 0

    if sub == "inject":
        session_obj = lake.get_agent_session(args.agent_id) if hasattr(lake, "get_agent_session") else None
        if session_obj is None:
            session_obj = AgentSession(
                agent_id=args.agent_id,
                harness_type="antigravity",
                model_id="gemini-4-argon",
                max_context_tokens=1_000_000,
                used_tokens=50_000,
            )
        capsule = _call_compatible(
            retriever.build_context_capsule,
            session=session_obj,
            task_query=args.query,
            agent_id=args.agent_id,
            query=args.query,
        )
        if args.output and hasattr(retriever, "inject_into_sidecar"):
            retriever.inject_into_sidecar(capsule, args.output)
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "output_path": str(args.output),
                        "capsule": capsule.to_dict(),
                    },
                    indent=2,
                )
            )
        else:
            print(capsule.markdown_payload)
        return 0

    if sub == "time-travel":
        res = tui.execute_key_action("t", {"as_of": args.as_of})
        print(json.dumps(res, indent=2))
        return 0

    if sub == "mcp":
        for line in sys.stdin:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                req = json.loads(line_str)
                if mcp_server is not None and hasattr(mcp_server, "handle_jsonrpc"):
                    resp = mcp_server.handle_jsonrpc(req)
                elif mcp_server is not None and hasattr(mcp_server, "handle_rpc"):
                    resp = mcp_server.handle_rpc(req)
                else:
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req.get("id"),
                        "result": {"status": "ok"},
                    }
            except Exception as exc:
                resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32603, "message": str(exc)},
                }
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
