"""
Federated Context Lake (SQLite Bitemporal Knowledge Graph + Vector Store).
Provides full ACID CRUD for ConceptNodes, immutable bitemporal BeliefRevisions,
RelationEdges, AgentSessions, ConflictResolutionEvents, and time-travel snapshots.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from context_palace.models import (
    AgentSession,
    BeliefRevision,
    ConceptNode,
    ConflictResolutionEvent,
    RelationEdge,
    compute_text_embedding,
)


class ContextLake:
    """
    SQLite-backed Bitemporal Knowledge Graph and Federated Context Lake.
    Supports both in-memory (`:memory:`) and persistent file-backed databases.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            p = Path(db_path).expanduser().resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(p)

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;

            CREATE TABLE IF NOT EXISTS agent_sessions (
                agent_id TEXT PRIMARY KEY,
                harness_type TEXT NOT NULL,
                model_id TEXT NOT NULL,
                max_context_tokens INTEGER NOT NULL,
                used_tokens INTEGER NOT NULL DEFAULT 0,
                transcript_path TEXT,
                last_processed_turn INTEGER NOT NULL DEFAULT 0,
                last_byte_offset INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL DEFAULT 0.0,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS concept_nodes (
                node_id TEXT PRIMARY KEY,
                canonical_name TEXT NOT NULL,
                category TEXT NOT NULL,
                active_value TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.9,
                pinned INTEGER NOT NULL DEFAULT 0,
                embedding_json TEXT NOT NULL DEFAULT '[]',
                source_agent_id TEXT NOT NULL DEFAULT 'system',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                tombstoned INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_concept_nodes_name
                ON concept_nodes(LOWER(canonical_name));

            CREATE TABLE IF NOT EXISTS belief_revisions (
                revision_id TEXT PRIMARY KEY,
                node_id TEXT NOT NULL,
                value TEXT NOT NULL,
                confidence REAL NOT NULL,
                source_agent_id TEXT NOT NULL,
                provenance_type TEXT NOT NULL DEFAULT 'agent_turn',
                turn_id INTEGER NOT NULL DEFAULT 0,
                valid_from REAL NOT NULL,
                valid_to REAL,
                superseded_by TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_belief_revisions_node
                ON belief_revisions(node_id, valid_from);

            CREATE TABLE IF NOT EXISTS relation_edges (
                edge_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                source_agent_id TEXT NOT NULL DEFAULT 'system',
                valid_from REAL NOT NULL,
                valid_to REAL
            );

            CREATE INDEX IF NOT EXISTS idx_relation_edges_src_tgt
                ON relation_edges(source_id, target_id);

            CREATE TABLE IF NOT EXISTS conflict_events (
                event_id TEXT PRIMARY KEY,
                node_id TEXT NOT NULL,
                winning_revision_id TEXT NOT NULL,
                losing_revision_id TEXT NOT NULL,
                winning_agent_id TEXT NOT NULL,
                losing_agent_id TEXT NOT NULL,
                winning_score REAL NOT NULL,
                losing_score REAL NOT NULL,
                reason TEXT NOT NULL,
                resolved_at REAL NOT NULL
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -------------------------------------------------------------------------
    # AgentSession CRUD
    # -------------------------------------------------------------------------

    def upsert_agent_session(self, session: AgentSession) -> AgentSession:
        now = session.updated_at if session.updated_at > 0 else time.time()
        session.updated_at = now
        self._conn.execute(
            """
            INSERT INTO agent_sessions (
                agent_id, harness_type, model_id, max_context_tokens,
                used_tokens, transcript_path, last_processed_turn,
                last_byte_offset, updated_at, metadata_json
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
        self._conn.commit()
        return session

    def get_agent_session(self, agent_id: str) -> Optional[AgentSession]:
        row = self._conn.execute(
            "SELECT * FROM agent_sessions WHERE agent_id = ?", (agent_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_agent_session(row)

    def list_agent_sessions(self) -> List[AgentSession]:
        rows = self._conn.execute(
            "SELECT * FROM agent_sessions ORDER BY updated_at DESC, agent_id ASC"
        ).fetchall()
        return [self._row_to_agent_session(r) for r in rows]

    @staticmethod
    def _row_to_agent_session(row: sqlite3.Row) -> AgentSession:
        try:
            meta = json.loads(row["metadata_json"] or "{}")
        except Exception:
            meta = {}
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
            metadata=meta,
        )

    # -------------------------------------------------------------------------
    # ConceptNode & BeliefRevision CRUD
    # -------------------------------------------------------------------------

    def create_concept(
        self,
        canonical_name: str,
        category: str,
        value: str,
        summary: str = "",
        confidence: float = 0.9,
        source_agent_id: str = "system",
        provenance_type: str = "operator_crud",
        turn_id: int = 0,
        pinned: bool = False,
        timestamp: Optional[float] = None,
        node_id: Optional[str] = None,
    ) -> ConceptNode:
        ts = timestamp if timestamp is not None else time.time()
        nid = node_id or f"node-{uuid.uuid4().hex[:10]}"
        summary_text = summary or f"{canonical_name}: {value[:120]}"
        emb = compute_text_embedding(f"{canonical_name} {value} {summary_text}")

        node = ConceptNode(
            node_id=nid,
            canonical_name=canonical_name.strip(),
            category=category.strip() or "entity",
            active_value=value,
            summary=summary_text,
            confidence=confidence,
            pinned=bool(pinned),
            embedding=emb,
            source_agent_id=source_agent_id,
            created_at=ts,
            updated_at=ts,
            tombstoned=False,
        )

        rev = BeliefRevision(
            revision_id=f"rev-{uuid.uuid4().hex[:10]}",
            node_id=nid,
            value=value,
            confidence=confidence,
            source_agent_id=source_agent_id,
            provenance_type=provenance_type,
            turn_id=turn_id,
            valid_from=ts,
            valid_to=None,
            superseded_by=None,
        )

        self._conn.execute(
            """
            INSERT INTO concept_nodes (
                node_id, canonical_name, category, active_value, summary,
                confidence, pinned, embedding_json, source_agent_id,
                created_at, updated_at, tombstoned
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                node.node_id,
                node.canonical_name,
                node.category,
                node.active_value,
                node.summary,
                node.confidence,
                1 if node.pinned else 0,
                json.dumps(node.embedding),
                node.source_agent_id,
                node.created_at,
                node.updated_at,
            ),
        )
        self.record_belief_revision(rev)
        self._conn.commit()
        return node

    def get_concept(self, node_id_or_name: str, include_tombstoned: bool = False) -> Optional[ConceptNode]:
        if not node_id_or_name:
            return None
        key = node_id_or_name.strip()
        tomb_clause = "" if include_tombstoned else "AND tombstoned = 0"
        row = self._conn.execute(
            f"""
            SELECT * FROM concept_nodes
            WHERE (node_id = ? OR LOWER(canonical_name) = LOWER(?))
            {tomb_clause}
            ORDER BY tombstoned ASC, updated_at DESC
            LIMIT 1
            """,
            (key, key),
        ).fetchone()
        if not row:
            return None
        return self._row_to_concept(row)

    def list_concepts(
        self,
        include_tombstoned: bool = False,
        category: Optional[str] = None,
        pinned_only: bool = False,
    ) -> List[ConceptNode]:
        clauses: List[str] = []
        params: List[Any] = []
        if not include_tombstoned:
            clauses.append("tombstoned = 0")
        if category:
            clauses.append("LOWER(category) = LOWER(?)")
            params.append(category.strip())
        if pinned_only:
            clauses.append("pinned = 1")

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM concept_nodes {where_sql} ORDER BY pinned DESC, updated_at DESC, canonical_name ASC",
            tuple(params),
        ).fetchall()
        return [self._row_to_concept(r) for r in rows]

    def update_concept(
        self,
        node_id_or_name: str,
        new_value: str,
        summary: Optional[str] = None,
        confidence: float = 0.95,
        source_agent_id: str = "operator",
        provenance_type: str = "operator_crud",
        turn_id: int = 0,
        pinned: Optional[bool] = None,
        timestamp: Optional[float] = None,
    ) -> ConceptNode:
        existing = self.get_concept(node_id_or_name, include_tombstoned=True)
        if existing is None:
            raise KeyError(f"Concept not found: {node_id_or_name}")

        ts = timestamp if timestamp is not None else time.time()
        new_summary = summary if summary is not None else f"{existing.canonical_name}: {new_value[:120]}"
        new_pinned = existing.pinned if pinned is None else bool(pinned)
        new_emb = compute_text_embedding(f"{existing.canonical_name} {new_value} {new_summary}")

        new_rev = BeliefRevision(
            revision_id=f"rev-{uuid.uuid4().hex[:10]}",
            node_id=existing.node_id,
            value=new_value,
            confidence=confidence,
            source_agent_id=source_agent_id,
            provenance_type=provenance_type,
            turn_id=turn_id,
            valid_from=ts,
            valid_to=None,
            superseded_by=None,
        )

        # Find active revision(s) for this node and supersede them
        active_rows = self._conn.execute(
            "SELECT revision_id FROM belief_revisions WHERE node_id = ? AND valid_to IS NULL",
            (existing.node_id,),
        ).fetchall()

        if active_rows:
            for r in active_rows:
                self._conn.execute(
                    """
                    UPDATE belief_revisions
                    SET valid_to = ?, superseded_by = ?
                    WHERE revision_id = ?
                    """,
                    (ts, new_rev.revision_id, r["revision_id"]),
                )
        self.record_belief_revision(new_rev)

        self._conn.execute(
            """
            UPDATE concept_nodes
            SET active_value = ?,
                summary = ?,
                confidence = ?,
                pinned = ?,
                embedding_json = ?,
                source_agent_id = ?,
                updated_at = ?,
                tombstoned = 0
            WHERE node_id = ?
            """,
            (
                new_value,
                new_summary,
                confidence,
                1 if new_pinned else 0,
                json.dumps(new_emb),
                source_agent_id,
                ts,
                existing.node_id,
            ),
        )
        self._conn.commit()
        updated = self.get_concept(existing.node_id, include_tombstoned=True)
        assert updated is not None
        return updated

    def delete_concept(
        self,
        node_id_or_name: str,
        hard: bool = False,
        timestamp: Optional[float] = None,
    ) -> bool:
        existing = self.get_concept(node_id_or_name, include_tombstoned=True)
        if existing is None:
            return False

        ts = timestamp if timestamp is not None else time.time()
        if hard:
            self._conn.execute("DELETE FROM belief_revisions WHERE node_id = ?", (existing.node_id,))
            self._conn.execute(
                "DELETE FROM relation_edges WHERE source_id = ? OR target_id = ?",
                (existing.node_id, existing.node_id),
            )
            self._conn.execute("DELETE FROM conflict_events WHERE node_id = ?", (existing.node_id,))
            self._conn.execute("DELETE FROM concept_nodes WHERE node_id = ?", (existing.node_id,))
        else:
            self._conn.execute(
                "UPDATE concept_nodes SET tombstoned = 1, updated_at = ? WHERE node_id = ?",
                (ts, existing.node_id),
            )
            self._conn.execute(
                "UPDATE belief_revisions SET valid_to = ? WHERE node_id = ? AND valid_to IS NULL",
                (ts, existing.node_id),
            )
            self._conn.execute(
                """
                UPDATE relation_edges
                SET valid_to = ?
                WHERE (source_id = ? OR target_id = ?) AND valid_to IS NULL
                """,
                (ts, existing.node_id, existing.node_id),
            )
        self._conn.commit()
        return True

    def pin_concept(self, node_id_or_name: str, pinned: bool = True) -> Optional[ConceptNode]:
        existing = self.get_concept(node_id_or_name, include_tombstoned=True)
        if existing is None:
            return None
        self._conn.execute(
            "UPDATE concept_nodes SET pinned = ? WHERE node_id = ?",
            (1 if pinned else 0, existing.node_id),
        )
        self._conn.commit()
        return self.get_concept(existing.node_id, include_tombstoned=True)

    @staticmethod
    def _row_to_concept(row: sqlite3.Row) -> ConceptNode:
        try:
            emb = json.loads(row["embedding_json"] or "[]")
        except Exception:
            emb = []
        return ConceptNode(
            node_id=row["node_id"],
            canonical_name=row["canonical_name"],
            category=row["category"],
            active_value=row["active_value"],
            summary=row["summary"],
            confidence=float(row["confidence"]),
            pinned=bool(row["pinned"]),
            embedding=emb,
            source_agent_id=row["source_agent_id"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            tombstoned=bool(row["tombstoned"]),
        )

    # -------------------------------------------------------------------------
    # Edges
    # -------------------------------------------------------------------------

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
        source_agent_id: str = "system",
        timestamp: Optional[float] = None,
    ) -> RelationEdge:
        ts = timestamp if timestamp is not None else time.time()
        # Allow passing canonical names or node_ids
        src_node = self.get_concept(source_id, include_tombstoned=True)
        tgt_node = self.get_concept(target_id, include_tombstoned=True)
        resolved_src = src_node.node_id if src_node else source_id
        resolved_tgt = tgt_node.node_id if tgt_node else target_id

        edge = RelationEdge(
            edge_id=f"edge-{uuid.uuid4().hex[:10]}",
            source_id=resolved_src,
            target_id=resolved_tgt,
            relation_type=relation_type,
            weight=weight,
            source_agent_id=source_agent_id,
            valid_from=ts,
            valid_to=None,
        )
        self._conn.execute(
            """
            INSERT INTO relation_edges (
                edge_id, source_id, target_id, relation_type,
                weight, source_agent_id, valid_from, valid_to
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                edge.edge_id,
                edge.source_id,
                edge.target_id,
                edge.relation_type,
                edge.weight,
                edge.source_agent_id,
                edge.valid_from,
                edge.valid_to,
            ),
        )
        self._conn.commit()
        return edge

    def list_edges(
        self,
        node_id: Optional[str] = None,
        active_only: bool = True,
    ) -> List[RelationEdge]:
        clauses: List[str] = []
        params: List[Any] = []
        if active_only:
            clauses.append("valid_to IS NULL")
        if node_id:
            concept = self.get_concept(node_id, include_tombstoned=True)
            resolved_id = concept.node_id if concept else node_id
            clauses.append("(source_id = ? OR target_id = ?)")
            params.extend([resolved_id, resolved_id])

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM relation_edges {where_sql} ORDER BY valid_from ASC",
            tuple(params),
        ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    @staticmethod
    def _row_to_edge(row: sqlite3.Row) -> RelationEdge:
        return RelationEdge(
            edge_id=row["edge_id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            relation_type=row["relation_type"],
            weight=float(row["weight"]),
            source_agent_id=row["source_agent_id"],
            valid_from=float(row["valid_from"]),
            valid_to=float(row["valid_to"]) if row["valid_to"] is not None else None,
        )

    # -------------------------------------------------------------------------
    # Belief Revisions & Conflict Events
    # -------------------------------------------------------------------------

    def get_belief_history(self, node_id: str) -> List[BeliefRevision]:
        concept = self.get_concept(node_id, include_tombstoned=True)
        resolved_id = concept.node_id if concept else node_id
        rows = self._conn.execute(
            "SELECT * FROM belief_revisions WHERE node_id = ? ORDER BY valid_from ASC, rowid ASC",
            (resolved_id,),
        ).fetchall()
        return [self._row_to_revision(r) for r in rows]

    def record_belief_revision(self, revision: BeliefRevision) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO belief_revisions (
                revision_id, node_id, value, confidence,
                source_agent_id, provenance_type, turn_id,
                valid_from, valid_to, superseded_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                revision.revision_id,
                revision.node_id,
                revision.value,
                revision.confidence,
                revision.source_agent_id,
                revision.provenance_type,
                revision.turn_id,
                revision.valid_from,
                revision.valid_to,
                revision.superseded_by,
            ),
        )
        self._conn.commit()

    def supersede_revision(
        self,
        old_revision_id: str,
        new_revision: BeliefRevision,
        valid_to: float,
    ) -> None:
        self._conn.execute(
            """
            UPDATE belief_revisions
            SET valid_to = ?, superseded_by = ?
            WHERE revision_id = ?
            """,
            (valid_to, new_revision.revision_id, old_revision_id),
        )
        self.record_belief_revision(new_revision)
        self._conn.commit()

    @staticmethod
    def _row_to_revision(row: sqlite3.Row) -> BeliefRevision:
        return BeliefRevision(
            revision_id=row["revision_id"],
            node_id=row["node_id"],
            value=row["value"],
            confidence=float(row["confidence"]),
            source_agent_id=row["source_agent_id"],
            provenance_type=row["provenance_type"],
            turn_id=int(row["turn_id"]),
            valid_from=float(row["valid_from"]),
            valid_to=float(row["valid_to"]) if row["valid_to"] is not None else None,
            superseded_by=row["superseded_by"],
        )

    def record_conflict_event(self, event: ConflictResolutionEvent) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO conflict_events (
                event_id, node_id, winning_revision_id, losing_revision_id,
                winning_agent_id, losing_agent_id, winning_score, losing_score,
                reason, resolved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.node_id,
                event.winning_revision_id,
                event.losing_revision_id,
                event.winning_agent_id,
                event.losing_agent_id,
                event.winning_score,
                event.losing_score,
                event.reason,
                event.resolved_at,
            ),
        )
        self._conn.commit()

    def list_conflict_events(self, node_id: Optional[str] = None) -> List[ConflictResolutionEvent]:
        if node_id:
            concept = self.get_concept(node_id, include_tombstoned=True)
            resolved_id = concept.node_id if concept else node_id
            rows = self._conn.execute(
                "SELECT * FROM conflict_events WHERE node_id = ? ORDER BY resolved_at DESC",
                (resolved_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
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

    # -------------------------------------------------------------------------
    # Bitemporal Time-Travel Snapshot
    # -------------------------------------------------------------------------

    def snapshot(self, as_of: float) -> Dict[str, Any]:
        """
        Reconstruct the exact knowledge graph state at timestamp `as_of`.
        Finds all BeliefRevision rows where `valid_from <= as_of AND (valid_to IS NULL OR valid_to > as_of)`,
        reconstructs the corresponding ConceptNode states at that timestamp, and includes all
        RelationEdge rows valid at `as_of`.
        """
        rev_rows = self._conn.execute(
            """
            SELECT * FROM belief_revisions
            WHERE valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)
            ORDER BY valid_from ASC, rowid ASC
            """,
            (as_of, as_of),
        ).fetchall()

        active_revisions = [self._row_to_revision(r) for r in rev_rows]
        latest_rev_by_node: Dict[str, BeliefRevision] = {}
        for rev in active_revisions:
            latest_rev_by_node[rev.node_id] = rev

        reconstructed_nodes: List[ConceptNode] = []
        for nid, rev in latest_rev_by_node.items():
            base_node = self.get_concept(nid, include_tombstoned=True)
            if base_node is None:
                continue
            summary_at_ts = f"{base_node.canonical_name}: {rev.value[:120]}"
            emb_at_ts = compute_text_embedding(f"{base_node.canonical_name} {rev.value} {summary_at_ts}")
            reconstructed_nodes.append(
                ConceptNode(
                    node_id=base_node.node_id,
                    canonical_name=base_node.canonical_name,
                    category=base_node.category,
                    active_value=rev.value,
                    summary=summary_at_ts,
                    confidence=rev.confidence,
                    pinned=base_node.pinned,
                    embedding=emb_at_ts,
                    source_agent_id=rev.source_agent_id,
                    created_at=base_node.created_at,
                    updated_at=rev.valid_from,
                    tombstoned=False,
                )
            )

        edge_rows = self._conn.execute(
            """
            SELECT * FROM relation_edges
            WHERE valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)
            ORDER BY valid_from ASC
            """,
            (as_of, as_of),
        ).fetchall()
        active_edges = [self._row_to_edge(r) for r in edge_rows]

        return {
            "as_of": as_of,
            "nodes": reconstructed_nodes,
            "concepts": reconstructed_nodes,
            "revisions": active_revisions,
            "edges": active_edges,
        }
