"""
Unit and integration tests for IncrementalDeltaExtractor and ConsensusEngine.
Uses an in-memory FakeContextLake matching the exact ContextLake method contracts.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import pytest

from context_palace.consensus import PROVENANCE_WEIGHTS, ConsensusEngine
from context_palace.extractor import IncrementalDeltaExtractor
from context_palace.models import (
    AgentSession,
    BeliefRevision,
    ConceptNode,
    ConflictResolutionEvent,
    RelationEdge,
    compute_text_embedding,
)


class FakeContextLake:
    """In-memory mock matching ContextLake method signatures."""

    def __init__(self) -> None:
        self.nodes: Dict[str, ConceptNode] = {}
        self.revisions: Dict[str, List[BeliefRevision]] = {}
        self.edges: List[RelationEdge] = []
        self.conflicts: List[ConflictResolutionEvent] = []
        self.sessions: Dict[str, AgentSession] = {}

    def list_concepts(self, include_tombstoned: bool = False) -> List[ConceptNode]:
        return [
            n for n in self.nodes.values() if include_tombstoned or not n.tombstoned
        ]

    def create_concept(
        self,
        canonical_name: str,
        category: str,
        active_value: str,
        summary: str = "",
        confidence: float = 0.9,
        pinned: bool = False,
        source_agent_id: str = "system",
        provenance_type: str = "agent_turn",
        turn_id: int = 0,
        node_id: Optional[str] = None,
        created_at: float = 1000.0,
    ) -> ConceptNode:
        nid = node_id or f"node_{uuid.uuid4().hex[:10]}"
        node = ConceptNode(
            node_id=nid,
            canonical_name=canonical_name,
            category=category,
            active_value=active_value,
            summary=summary or f"{canonical_name}: {active_value}",
            confidence=confidence,
            pinned=pinned,
            embedding=compute_text_embedding(f"{canonical_name} {active_value}"),
            source_agent_id=source_agent_id,
            created_at=created_at,
            updated_at=created_at,
            tombstoned=False,
        )
        self.nodes[nid] = node
        rev = BeliefRevision(
            revision_id=f"rev_{uuid.uuid4().hex[:10]}",
            node_id=nid,
            value=active_value,
            confidence=confidence,
            source_agent_id=source_agent_id,
            provenance_type=provenance_type,
            turn_id=turn_id,
            valid_from=created_at,
            valid_to=None,
        )
        self.revisions.setdefault(nid, []).append(rev)
        return node

    def update_concept(
        self,
        node_id: str,
        active_value: str,
        summary: Optional[str] = None,
        confidence: Optional[float] = None,
        source_agent_id: str = "system",
        provenance_type: str = "agent_turn",
        turn_id: int = 0,
        updated_at: float = 2000.0,
    ) -> ConceptNode:
        node = self.nodes[node_id]
        new_rev_id = f"rev_{uuid.uuid4().hex[:10]}"
        for rev in self.revisions.get(node_id, []):
            if rev.valid_to is None:
                rev.valid_to = updated_at
                rev.superseded_by = new_rev_id

        node.active_value = active_value
        if summary is not None:
            node.summary = summary
        if confidence is not None:
            node.confidence = confidence
        node.source_agent_id = source_agent_id
        node.updated_at = updated_at
        node.embedding = compute_text_embedding(f"{node.canonical_name} {active_value}")

        new_rev = BeliefRevision(
            revision_id=new_rev_id,
            node_id=node_id,
            value=active_value,
            confidence=node.confidence,
            source_agent_id=source_agent_id,
            provenance_type=provenance_type,
            turn_id=turn_id,
            valid_from=updated_at,
            valid_to=None,
        )
        self.revisions.setdefault(node_id, []).append(new_rev)
        return node

    def get_belief_history(self, node_id: str) -> List[BeliefRevision]:
        return list(self.revisions.get(node_id, []))

    def record_conflict_event(self, event: ConflictResolutionEvent) -> None:
        self.conflicts.append(event)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
        source_agent_id: str = "system",
        valid_from: float = 1000.0,
    ) -> RelationEdge:
        edge = RelationEdge(
            edge_id=f"edge_{uuid.uuid4().hex[:10]}",
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            source_agent_id=source_agent_id,
            valid_from=valid_from,
        )
        self.edges.append(edge)
        return edge

    def upsert_agent_session(self, session: AgentSession) -> AgentSession:
        self.sessions[session.agent_id] = session
        return session


def test_provenance_weights_hierarchy() -> None:
    """Verify PROVENANCE_WEIGHTS strictly orders operator_crud > user_explicit > verified_test > agent_turn > model_inference."""
    engine = ConsensusEngine()
    ordered_types = [
        "operator_crud",
        "user_explicit",
        "verified_test",
        "agent_turn",
        "model_inference",
    ]
    scores = []
    for ptype in ordered_types:
        assert ptype in PROVENANCE_WEIGHTS
        rev = BeliefRevision(
            revision_id=f"rev_{ptype}",
            node_id="node_1",
            value="v1",
            confidence=0.90,
            source_agent_id="agent_a",
            provenance_type=ptype,
            turn_id=1,
            valid_from=1000.0,
        )
        scores.append(engine.score_revision(rev, current_time=1000.0, vote_count=1))

    for i in range(len(scores) - 1):
        assert scores[i] > scores[i + 1], f"Expected {ordered_types[i]} > {ordered_types[i + 1]}"


def test_consensus_recency_and_vote_bonus() -> None:
    """Verify newer revisions score higher than older ones and multi-agent votes boost consensus score."""
    engine = ConsensusEngine()
    older = BeliefRevision(
        revision_id="rev_old",
        node_id="node_db",
        value="PostgreSQL 14",
        confidence=0.85,
        source_agent_id="agent_1",
        provenance_type="agent_turn",
        turn_id=1,
        valid_from=1000.0,
    )
    newer = BeliefRevision(
        revision_id="rev_new",
        node_id="node_db",
        value="PostgreSQL 16",
        confidence=0.85,
        source_agent_id="agent_2",
        provenance_type="agent_turn",
        turn_id=2,
        valid_from=2500.0,
    )
    assert engine.score_revision(newer, current_time=2500.0) > engine.score_revision(older, current_time=2500.0)
    assert engine.score_revision(newer) > engine.score_revision(older)

    single_vote = engine.score_revision(older, current_time=2500.0, vote_count=1)
    three_votes = engine.score_revision(older, current_time=2500.0, vote_count=3)
    capped_votes = engine.score_revision(older, current_time=2500.0, vote_count=10)
    assert round(three_votes - single_vote, 4) == 0.16
    assert round(capped_votes - single_vote, 4) == 0.25


def test_consensus_resolve_authority_over_agent_turn() -> None:
    """Verify user_explicit and operator_crud revisions defeat conflicting lower-authority agent_turn updates."""
    engine = ConsensusEngine()
    user_rev = BeliefRevision(
        revision_id="rev_user",
        node_id="node_port",
        value="8080",
        confidence=0.95,
        source_agent_id="user_main",
        provenance_type="user_explicit",
        turn_id=1,
        valid_from=1000.0,
    )
    agent_rev = BeliefRevision(
        revision_id="rev_agent",
        node_id="node_port",
        value="3000",
        confidence=0.85,
        source_agent_id="worker_agent",
        provenance_type="agent_turn",
        turn_id=2,
        valid_from=1200.0,
    )
    winner, loser, event = engine.resolve(user_rev, agent_rev, resolved_at=1200.0)
    assert winner.revision_id == "rev_user"
    assert loser.revision_id == "rev_agent"
    assert event.winning_revision_id == "rev_user"
    assert event.losing_revision_id == "rev_agent"
    assert event.winning_score > event.losing_score
    assert "provenance authority" in event.reason


def test_consensus_resolve_multi_agent_vote_override() -> None:
    """Verify multi-agent consensus votes allow a corroborated revision to beat a single-agent belief."""
    engine = ConsensusEngine()
    existing = BeliefRevision(
        revision_id="rev_solo",
        node_id="node_cache",
        value="Memcached",
        confidence=0.88,
        source_agent_id="agent_solo",
        provenance_type="agent_turn",
        turn_id=1,
        valid_from=1000.0,
    )
    candidate = BeliefRevision(
        revision_id="rev_quorum",
        node_id="node_cache",
        value="Redis Cluster",
        confidence=0.85,
        source_agent_id="agent_swarm",
        provenance_type="agent_turn",
        turn_id=2,
        valid_from=1000.0,
    )
    winner, loser, event = engine.resolve(
        existing,
        candidate,
        vote_counts={"rev_solo": 1, "rev_quorum": 4},
        resolved_at=1000.0,
    )
    assert winner.revision_id == "rev_quorum"
    assert loser.revision_id == "rev_solo"
    assert "multi-agent consensus votes" in event.reason


def test_consensus_tie_breaker_on_equal_timestamp_and_provenance() -> None:
    """Verify candidate_revision wins tie-breaker when timestamp, provenance, and confidence are equal."""
    engine = ConsensusEngine()
    existing = BeliefRevision(
        revision_id="rev_1",
        node_id="node_model",
        value="claude-3-5-sonnet",
        confidence=0.85,
        source_agent_id="agent_1",
        provenance_type="agent_turn",
        turn_id=1,
        valid_from=1500.0,
    )
    candidate = BeliefRevision(
        revision_id="rev_2",
        node_id="node_model",
        value="claude-sonnet-4-5",
        confidence=0.85,
        source_agent_id="agent_2",
        provenance_type="agent_turn",
        turn_id=2,
        valid_from=1500.0,
    )
    winner, loser, event = engine.resolve(existing, candidate, resolved_at=1500.0)
    assert winner.revision_id == "rev_2"
    assert loser.revision_id == "rev_1"
    assert event.winning_score > event.losing_score


def test_extract_from_text_explicit_kv_and_markdown() -> None:
    """Verify extract_from_text parses Key: Value, **Key**: Value, key = value, and constraints."""
    extractor = IncrementalDeltaExtractor()
    text = (
        "**Database**: PostgreSQL 16\n"
        "api_port = 8080\n"
        "Model is claude-sonnet-4-5\n"
        "Constraint: Never expose raw API keys in logs\n"
    )
    nodes, edges = extractor.extract_from_text(
        text, source_agent_id="agent_alpha", provenance_type="user_explicit", turn_id=1, timestamp=1000.0
    )
    by_name = {n.canonical_name: n for n in nodes}
    assert "database" in by_name
    assert by_name["database"].active_value == "PostgreSQL 16"
    assert by_name["database"].category == "architecture"

    assert "api_port" in by_name
    assert by_name["api_port"].active_value == "8080"
    assert by_name["api_port"].category == "config"

    assert "model" in by_name
    assert by_name["model"].active_value == "claude-sonnet-4-5"
    assert by_name["model"].category == "config"

    constraint_nodes = [n for n in nodes if n.category == "constraint"]
    assert len(constraint_nodes) == 1
    assert "Never expose raw API keys" in constraint_nodes[0].active_value
    assert len(edges) >= 1


def test_extract_from_text_architectural_relations() -> None:
    """Verify architectural statements ('X uses Y', 'X depends on Y', 'X connects to Y') emit nodes and edges."""
    extractor = IncrementalDeltaExtractor()
    text = "API Gateway uses Redis Cache\nWorker Service depends on PostgreSQL"
    nodes, edges = extractor.extract_from_text(text, source_agent_id="architect", turn_id=2, timestamp=1100.0)
    names = {n.canonical_name for n in nodes}
    assert "api_gateway" in names
    assert "redis_cache" in names
    assert "worker_service" in names
    assert "postgresql" in names
    assert any(e.relation_type == "depends_on" for e in edges)


def test_find_matching_concept_vector_and_canonical_dedup() -> None:
    """Verify find_matching_concept deduplicates equivalent concepts while keeping distinct attributes separate."""
    extractor = IncrementalDeltaExtractor(similarity_threshold=0.78)
    db_node = ConceptNode(
        node_id="node_db",
        canonical_name="database",
        category="architecture",
        active_value="PostgreSQL",
        embedding=compute_text_embedding("database PostgreSQL"),
    )
    host_node = ConceptNode(
        node_id="node_db_host",
        canonical_name="database_host",
        category="config",
        active_value="localhost",
        embedding=compute_text_embedding("database_host localhost"),
    )
    existing = [db_node, host_node]

    # Synonym & prefix variation should match 'database'
    match_syn = extractor.find_matching_concept("Primary Database Engine", "PostgreSQL", existing)
    assert match_syn is not None
    assert match_syn.node_id == "node_db"

    # Substring/token overlap should match 'database_host'
    match_host = extractor.find_matching_concept("primary_database_host", "127.0.0.1", existing)
    assert match_host is not None
    assert match_host.node_id == "node_db_host"

    # Sibling config attribute 'database_port' must NOT falsely match 'database_host'
    match_port = extractor.find_matching_concept("database_port", "5432", [host_node])
    assert match_port is None


def test_ingest_turns_incremental_watermarking() -> None:
    """
    Verify running ingest_turns twice on the same transcript processes 0 turns on the second call,
    and only processes delta turns when new turns are appended.
    """
    lake = FakeContextLake()
    extractor = IncrementalDeltaExtractor()
    session = AgentSession(
        agent_id="claude_1",
        harness_type="claude_code",
        model_id="claude-sonnet-4-5",
        max_context_tokens=200000,
        used_tokens=100,
        last_processed_turn=0,
    )

    initial_turns = [
        {"turn_id": 1, "role": "user", "content": "Database: PostgreSQL 16\nPort: 8080"},
        {"turn_id": 2, "role": "assistant", "content": "Cache Backend: Redis 7"},
    ]

    first_res = extractor.ingest_turns(lake, session, initial_turns, timestamp=1000.0)
    assert first_res["processed_turns"] == 2
    assert first_res["skipped_turns"] == 0
    assert first_res["created_nodes"] == 3
    assert session.last_processed_turn == 2
    tokens_after_first = session.used_tokens
    assert tokens_after_first > 100

    # Re-running on identical transcript must process 0 turns
    second_res = extractor.ingest_turns(lake, session, initial_turns, timestamp=1050.0)
    assert second_res["processed_turns"] == 0
    assert second_res["skipped_turns"] == 2
    assert second_res["created_nodes"] == 0
    assert session.last_processed_turn == 2
    assert session.used_tokens == tokens_after_first

    # Appending turn 3 must process only the 1 delta turn
    extended_turns = initial_turns + [
        {"turn_id": 3, "role": "assistant", "content": "Queue Broker: RabbitMQ"},
    ]
    third_res = extractor.ingest_turns(lake, session, extended_turns, timestamp=1100.0)
    assert third_res["processed_turns"] == 1
    assert third_res["skipped_turns"] == 2
    assert third_res["created_nodes"] == 1
    assert session.last_processed_turn == 3
    assert session.used_tokens > tokens_after_first


def test_ingest_turns_merges_equivalent_concepts_across_turns() -> None:
    """Verify repeating equivalent concept values across turns increments merged_nodes without duplicate nodes."""
    lake = FakeContextLake()
    extractor = IncrementalDeltaExtractor()
    session = AgentSession(
        agent_id="agent_dedup",
        harness_type="antigravity",
        model_id="gemini-2.5-pro",
        max_context_tokens=1000000,
    )

    turns = [
        {"turn_id": 1, "role": "user", "content": "Database: PostgreSQL 16"},
        {"turn_id": 2, "role": "assistant", "content": "The primary database is postgresql 16"},
        {"turn_id": 3, "role": "assistant", "content": "**Database Engine**: PostgreSQL 16"},
    ]
    res = extractor.ingest_turns(lake, session, turns, timestamp=2000.0)
    assert res["processed_turns"] == 3
    assert res["created_nodes"] == 1
    assert res["merged_nodes"] == 2
    assert res["conflicts_resolved"] == 0
    assert len(lake.list_concepts()) == 1


def test_ingest_turns_conflict_resolution_protects_user_explicit_from_agent_override() -> None:
    """Verify user_explicit belief is preserved when a lower-authority agent_turn attempts a conflicting update."""
    lake = FakeContextLake()
    extractor = IncrementalDeltaExtractor()
    session = AgentSession(
        agent_id="agent_worker",
        harness_type="claude_code",
        model_id="claude-sonnet-4-5",
        max_context_tokens=200000,
    )

    turns = [
        {"turn_id": 1, "role": "user", "content": "Port: 8080", "timestamp": 1000.0},
        {"turn_id": 2, "role": "assistant", "content": "Port: 9090", "timestamp": 1010.0},
    ]
    res = extractor.ingest_turns(lake, session, turns, timestamp=1010.0)
    assert res["conflicts_resolved"] == 1
    concepts = lake.list_concepts()
    assert len(concepts) == 1
    assert concepts[0].active_value == "8080"
    assert len(lake.conflicts) == 1
    assert lake.conflicts[0].winning_score > lake.conflicts[0].losing_score


def test_ingest_turns_operator_crud_overrides_prior_agent_turn() -> None:
    """Verify higher-authority operator_crud or user_explicit turn overrides an earlier agent_turn belief."""
    lake = FakeContextLake()
    extractor = IncrementalDeltaExtractor()
    session = AgentSession(
        agent_id="agent_worker",
        harness_type="codex",
        model_id="gpt-4.1",
        max_context_tokens=128000,
    )

    turns = [
        {"turn_id": 1, "role": "assistant", "content": "Database: SQLite", "timestamp": 1000.0},
        {"turn_id": 2, "role": "operator_crud", "content": "Database: PostgreSQL 16", "timestamp": 1100.0},
    ]
    res = extractor.ingest_turns(lake, session, turns, timestamp=1100.0)
    assert res["created_nodes"] == 1
    assert res["conflicts_resolved"] == 1
    concepts = lake.list_concepts()
    assert len(concepts) == 1
    assert concepts[0].active_value == "PostgreSQL 16"
    history = lake.get_belief_history(concepts[0].node_id)
    assert len(history) == 2
    assert history[0].valid_to is not None
    assert history[1].is_active is True
    assert history[1].value == "PostgreSQL 16"
