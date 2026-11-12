"""
Comprehensive test suite for Worker Alpha:
- ModelContextRegistry: dynamic context window detection across Gemini (1M/2M),
  Claude (200K/1M), OpenAI (128K/1M), Grok, DeepSeek, Ollama/MLX (32K), and
  transcript parsing (Antigravity `USER_SETTINGS_CHANGE`, Claude Code `.jsonl`).
- ContextLake: SQLite bitemporal knowledge graph CRUD, belief revision chains,
  edges, conflict audit events, agent session persistence, and `snapshot(as_of=...)`
  time-travel queries.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from context_palace.lake import ContextLake
from context_palace.model_registry import BUILTIN_MODEL_SPECS, ModelContextRegistry
from context_palace.models import AgentSession, BeliefRevision, ConflictResolutionEvent


def test_builtin_catalog_completeness() -> None:
    """Verify required models exist in BUILTIN_MODEL_SPECS with accurate capacities."""
    assert BUILTIN_MODEL_SPECS["gemini-2.5-pro"].max_context_tokens == 2_000_000
    assert BUILTIN_MODEL_SPECS["gemini-1.5-pro"].max_context_tokens == 2_000_000
    assert BUILTIN_MODEL_SPECS["gemini-2.5-flash"].max_context_tokens == 1_000_000
    assert BUILTIN_MODEL_SPECS["gemini-4-argon"].max_context_tokens == 1_000_000
    assert BUILTIN_MODEL_SPECS["claude-sonnet-4-5-1m"].max_context_tokens == 1_000_000
    assert BUILTIN_MODEL_SPECS["claude-opus-4-6-1m"].max_context_tokens == 1_000_000
    assert BUILTIN_MODEL_SPECS["claude-sonnet-4-5"].max_context_tokens == 200_000
    assert BUILTIN_MODEL_SPECS["claude-opus-4"].max_context_tokens == 200_000
    assert BUILTIN_MODEL_SPECS["claude-3-7-sonnet"].max_context_tokens == 200_000
    assert BUILTIN_MODEL_SPECS["claude-3-5-sonnet"].max_context_tokens == 200_000
    assert BUILTIN_MODEL_SPECS["gpt-4.1"].max_context_tokens == 1_000_000
    assert BUILTIN_MODEL_SPECS["gpt-4o"].max_context_tokens == 128_000
    assert BUILTIN_MODEL_SPECS["o3"].max_context_tokens == 200_000
    assert BUILTIN_MODEL_SPECS["grok-3"].max_context_tokens == 131_072
    assert BUILTIN_MODEL_SPECS["grok-4"].max_context_tokens == 256_000
    assert BUILTIN_MODEL_SPECS["deepseek-r1"].max_context_tokens == 128_000
    assert BUILTIN_MODEL_SPECS["qwen2.5-coder:32b"].max_context_tokens == 32_768
    assert BUILTIN_MODEL_SPECS["llama3.3:70b"].max_context_tokens == 131_072
    assert BUILTIN_MODEL_SPECS["mlx-community/qwen2.5-7b"].max_context_tokens == 32_768


def test_resolve_model_window_normalization() -> None:
    """Verify normalization of human-readable names, date stamps, provider prefixes, and 1M tags."""
    reg = ModelContextRegistry()

    argon = reg.resolve_model_window("Gemini 4 Argon (Medium)")
    assert argon.model_id == "gemini-4-argon"
    assert argon.max_context_tokens == 1_000_000
    assert argon.provider == "google"

    sonnet_dated = reg.resolve_model_window("claude-sonnet-4-5-20250929")
    assert sonnet_dated.model_id == "claude-sonnet-4-5"
    assert sonnet_dated.max_context_tokens == 200_000
    assert sonnet_dated.provider == "anthropic"

    sonnet_prefixed = reg.resolve_model_window("anthropic/claude-3.5-sonnet")
    assert sonnet_prefixed.model_id == "claude-3-5-sonnet"
    assert sonnet_prefixed.max_context_tokens == 200_000

    opus_1m = reg.resolve_model_window("claude-opus-4-6[1m]")
    assert opus_1m.max_context_tokens == 1_000_000
    assert opus_1m.provider == "anthropic"

    mlx_spec = reg.resolve_model_window("mlx-community/qwen2.5-7b")
    assert mlx_spec.max_context_tokens == 32_768
    assert mlx_spec.provider == "mlx"

    unknown_spec = reg.resolve_model_window("mystery-custom-llm-v1")
    assert unknown_spec.max_context_tokens == 128_000


def test_detect_antigravity_session_from_jsonl(tmp_path: Path) -> None:
    """Verify parsing of synthetic Antigravity transcript.jsonl including USER_SETTINGS_CHANGE."""
    brain_dir = tmp_path / "brain" / "abcd1234-5678" / ".system_generated" / "logs"
    brain_dir.mkdir(parents=True)
    transcript = brain_dir / "transcript.jsonl"

    lines = [
        json.dumps(
            {
                "step_index": 0,
                "type": "USER_SETTINGS_CHANGE",
                "content": "Model Selection changed from Gemini 2.5 Flash to Gemini 4 Argon (Medium)",
            }
        ),
        json.dumps(
            {
                "step_index": 1,
                "type": "USER_INPUT",
                "content": "Build the Federated Context Lake and TUI CRUD Memory Palace.",
            }
        ),
        json.dumps(
            {
                "step_index": 2,
                "type": "PLANNER_RESPONSE",
                "content": "Initializing SQLite bitemporal schema and dynamic model context detector.",
            }
        ),
    ]
    transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")

    reg = ModelContextRegistry()
    session = reg.detect_agent_session_from_file(str(transcript))

    assert session.harness_type == "antigravity"
    assert session.model_id == "gemini-4-argon"
    assert session.max_context_tokens == 1_000_000
    assert session.used_tokens > 0
    assert session.last_processed_turn == 3
    assert session.last_byte_offset == transcript.stat().st_size
    assert session.remaining_headroom == 1_000_000 - session.used_tokens
    # With nearly 1M remaining headroom, adaptive_capsule_budget caps at max_cap (16384)
    assert session.adaptive_capsule_budget() == 16384


def test_detect_claude_code_session_and_discover(tmp_path: Path) -> None:
    """Verify parsing of Claude Code JSONL transcripts and discover_local_agents."""
    claude_dir = tmp_path / ".claude" / "projects" / "context-palace"
    claude_dir.mkdir(parents=True)
    session_file = claude_dir / "session-99.jsonl"

    lines = [
        json.dumps(
            {
                "sessionId": "sess-99",
                "cwd": "/Users/test/dev/context-palace",
                "type": "user",
                "message": {"role": "user", "content": "Inspect the context lake."},
            }
        ),
        json.dumps(
            {
                "sessionId": "sess-99",
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4-5-20250929",
                    "content": "Checking SQLite tables now.",
                },
            }
        ),
    ]
    session_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    reg = ModelContextRegistry()
    sess = reg.detect_agent_session_from_file(str(session_file), agent_id="claude-worker-1")
    assert sess.agent_id == "claude-worker-1"
    assert sess.harness_type == "claude_code"
    assert sess.model_id == "claude-sonnet-4-5"
    assert sess.max_context_tokens == 200_000
    assert sess.last_processed_turn == 2

    discovered = reg.discover_local_agents(search_dirs=[str(tmp_path)], limit=5)
    assert len(discovered) == 1
    assert discovered[0].model_id == "claude-sonnet-4-5"


def test_adaptive_capsule_budget_scaling() -> None:
    """Verify adaptive capsule budget scales down for tight 32K local models vs 1M cloud models."""
    local_sess = AgentSession(
        agent_id="mlx-local",
        harness_type="ollama",
        model_id="mlx-community/qwen2.5-7b",
        max_context_tokens=32_768,
        used_tokens=28_000,  # 4,768 tokens left
    )
    assert local_sess.remaining_headroom == 4_768
    assert local_sess.needs_compaction is True
    # 12% of 4768 = 572 tokens
    assert local_sess.adaptive_capsule_budget() == 572

    cloud_sess = AgentSession(
        agent_id="gemini-cloud",
        harness_type="antigravity",
        model_id="gemini-2.5-pro",
        max_context_tokens=2_000_000,
        used_tokens=50_000,
    )
    assert cloud_sess.needs_compaction is False
    assert cloud_sess.adaptive_capsule_budget() == 16_384


def test_lake_agent_session_crud() -> None:
    """Verify upserting, fetching, and listing AgentSession records in ContextLake."""
    lake = ContextLake(":memory:")
    s1 = AgentSession(
        agent_id="agent-alpha",
        harness_type="antigravity",
        model_id="gemini-4-argon",
        max_context_tokens=1_000_000,
        used_tokens=12_500,
        transcript_path="/tmp/t1.jsonl",
        last_processed_turn=14,
        last_byte_offset=48000,
        updated_at=1000.0,
        metadata={"branch": "swarm/model-lake"},
    )
    lake.upsert_agent_session(s1)

    fetched = lake.get_agent_session("agent-alpha")
    assert fetched is not None
    assert fetched.model_id == "gemini-4-argon"
    assert fetched.used_tokens == 12_500
    assert fetched.metadata["branch"] == "swarm/model-lake"

    # Update used_tokens via upsert
    s1.used_tokens = 25_000
    s1.updated_at = 1050.0
    lake.upsert_agent_session(s1)
    assert lake.get_agent_session("agent-alpha").used_tokens == 25_000
    assert len(lake.list_agent_sessions()) == 1


def test_lake_concept_crud_and_pinning() -> None:
    """Verify ConceptNode creation, case-insensitive lookup, filtering, and pinning."""
    lake = ContextLake(":memory:")
    n1 = lake.create_concept(
        canonical_name="Auth Token Strategy",
        category="architecture",
        value="Use Ed25519 signed JWTs with 15m TTL",
        summary="Ed25519 JWT auth with 15m TTL",
        confidence=0.92,
        source_agent_id="claude-1",
        provenance_type="user_explicit",
        turn_id=1,
        pinned=False,
        timestamp=100.0,
    )
    assert n1.node_id.startswith("node-")
    assert len(n1.embedding) == 64

    # Lookup by ID and by case-insensitive canonical_name
    by_id = lake.get_concept(n1.node_id)
    by_name = lake.get_concept("auth token strategy")
    assert by_id is not None and by_name is not None
    assert by_id.node_id == by_name.node_id

    # Pin concept
    pinned_node = lake.pin_concept("Auth Token Strategy", pinned=True)
    assert pinned_node is not None and pinned_node.pinned is True
    assert len(lake.list_concepts(pinned_only=True)) == 1
    assert len(lake.list_concepts(category="architecture")) == 1
    assert len(lake.list_concepts(category="nonexistent")) == 0


def test_lake_belief_revision_chain_on_update() -> None:
    """Verify updating a concept supersedes the previous active BeliefRevision and appends a new one."""
    lake = ContextLake(":memory:")
    node = lake.create_concept(
        canonical_name="Database Backend",
        category="config",
        value="PostgreSQL 15",
        confidence=0.80,
        source_agent_id="agent-1",
        timestamp=1000.0,
    )

    updated = lake.update_concept(
        node_id_or_name="Database Backend",
        new_value="SQLite WAL Bitemporal Lake",
        confidence=0.98,
        source_agent_id="operator",
        provenance_type="operator_crud",
        turn_id=5,
        timestamp=2000.0,
    )
    assert updated.active_value == "SQLite WAL Bitemporal Lake"
    assert updated.confidence == 0.98

    history = lake.get_belief_history(node.node_id)
    assert len(history) == 2
    rev1, rev2 = history[0], history[1]
    assert rev1.value == "PostgreSQL 15"
    assert rev1.valid_from == 1000.0
    assert rev1.valid_to == 2000.0
    assert rev1.superseded_by == rev2.revision_id
    assert rev1.is_active is False

    assert rev2.value == "SQLite WAL Bitemporal Lake"
    assert rev2.valid_from == 2000.0
    assert rev2.valid_to is None
    assert rev2.is_active is True


def test_lake_soft_and_hard_delete() -> None:
    """Verify soft delete (tombstoning + closing active revision) vs hard delete."""
    lake = ContextLake(":memory:")
    n1 = lake.create_concept(
        canonical_name="Legacy Cache",
        category="config",
        value="Redis 6",
        timestamp=100.0,
    )
    # Soft delete
    assert lake.delete_concept("Legacy Cache", hard=False, timestamp=200.0) is True
    assert lake.get_concept("Legacy Cache") is None
    assert len(lake.list_concepts(include_tombstoned=False)) == 0
    assert len(lake.list_concepts(include_tombstoned=True)) == 1

    hist = lake.get_belief_history(n1.node_id)
    assert len(hist) == 1
    assert hist[0].valid_to == 200.0

    # Hard delete
    assert lake.delete_concept(n1.node_id, hard=True) is True
    assert len(lake.list_concepts(include_tombstoned=True)) == 0
    assert len(lake.get_belief_history(n1.node_id)) == 0


def test_lake_edges_and_conflict_events() -> None:
    """Verify RelationEdge creation/filtering and ConflictResolutionEvent audit logging."""
    lake = ContextLake(":memory:")
    a = lake.create_concept("API Gateway", "architecture", "FastAPI on port 8080", timestamp=10.0)
    b = lake.create_concept("Context Lake DB", "architecture", "SQLite WAL", timestamp=10.0)

    edge = lake.add_edge(a.node_id, b.node_id, relation_type="depends_on", weight=0.95, timestamp=15.0)
    assert edge.source_id == a.node_id
    assert edge.target_id == b.node_id

    edges_for_a = lake.list_edges(node_id=a.node_id)
    assert len(edges_for_a) == 1
    assert edges_for_a[0].relation_type == "depends_on"

    ev = ConflictResolutionEvent(
        event_id="evt-001",
        node_id=a.node_id,
        winning_revision_id="rev-win",
        losing_revision_id="rev-lose",
        winning_agent_id="operator",
        losing_agent_id="stale-agent",
        winning_score=0.99,
        losing_score=0.52,
        reason="operator_crud overrides model_inference",
        resolved_at=25.0,
    )
    lake.record_conflict_event(ev)
    events = lake.list_conflict_events(node_id=a.node_id)
    assert len(events) == 1
    assert events[0].winning_agent_id == "operator"


def test_bitemporal_snapshot_time_travel() -> None:
    """Verify snapshot(as_of=...) reconstructs exact historical graph state across revisions and deletions."""
    lake = ContextLake(":memory:")
    c1 = lake.create_concept(
        canonical_name="Max Worker Count",
        category="config",
        value="4 workers",
        confidence=0.9,
        source_agent_id="init",
        timestamp=100.0,
    )
    c2 = lake.create_concept(
        canonical_name="Temp Feature Flag",
        category="config",
        value="enabled",
        confidence=0.8,
        source_agent_id="init",
        timestamp=100.0,
    )
    lake.add_edge(c1.node_id, c2.node_id, "configures", timestamp=100.0)

    # At t=200, update Max Worker Count to 8 workers
    lake.update_concept(
        "Max Worker Count",
        new_value="8 workers",
        confidence=0.95,
        source_agent_id="operator",
        timestamp=200.0,
    )
    # At t=300, soft-delete Temp Feature Flag
    lake.delete_concept("Temp Feature Flag", hard=False, timestamp=300.0)

    # Snapshot at t=150: both concepts active, Max Worker Count == "4 workers", 1 active edge
    snap_150 = lake.snapshot(as_of=150.0)
    nodes_150 = {n.canonical_name: n for n in snap_150["nodes"]}
    assert len(nodes_150) == 2
    assert nodes_150["Max Worker Count"].active_value == "4 workers"
    assert nodes_150["Temp Feature Flag"].active_value == "enabled"
    assert len(snap_150["edges"]) == 1

    # Snapshot at t=250: Max Worker Count == "8 workers", Temp Feature Flag still active
    snap_250 = lake.snapshot(as_of=250.0)
    nodes_250 = {n.canonical_name: n for n in snap_250["nodes"]}
    assert len(nodes_250) == 2
    assert nodes_250["Max Worker Count"].active_value == "8 workers"

    # Snapshot at t=350: Temp Feature Flag was tombstoned at t=300, so only Max Worker Count remains
    snap_350 = lake.snapshot(as_of=350.0)
    nodes_350 = {n.canonical_name: n for n in snap_350["nodes"]}
    assert len(nodes_350) == 1
    assert "Temp Feature Flag" not in nodes_350
    assert nodes_350["Max Worker Count"].active_value == "8 workers"
    assert len(snap_350["edges"]) == 0
