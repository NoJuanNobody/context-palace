"""
Unit & Integration Tests for Worker Delta: MemoryPalaceTUI and CLI (`context_palace.tui` & `context_palace.cli`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pytest

from context_palace.cli import _load_components, main
from context_palace.models import AgentSession, ConflictResolutionEvent, RelationEdge
from context_palace.tui import MemoryPalaceTUI


@pytest.fixture
def temp_db(tmp_path: Path) -> str:
    return str(tmp_path / "test_palace.db")


def test_tui_render_dashboard_shows_model_windows_and_compaction_warnings(temp_db: str) -> None:
    """Verify render_dashboard displays auto-detected model context windows, utilization %, COMPACT NOW, and concepts."""
    comps = _load_components(temp_db)
    lake = comps["lake"]
    tui: MemoryPalaceTUI = comps["tui"]

    # Register a healthy session (25% used) and a critical session (85% used -> COMPACT NOW)
    s_healthy = AgentSession(
        agent_id="agy-alpha",
        harness_type="antigravity",
        model_id="gemini-2.5-pro",
        max_context_tokens=1_000_000,
        used_tokens=250_000,
    )
    s_critical = AgentSession(
        agent_id="claude-beta",
        harness_type="claude_code",
        model_id="claude-sonnet-4",
        max_context_tokens=200_000,
        used_tokens=172_000,  # 86.0% -> COMPACT NOW
    )
    lake.upsert_agent_session(s_healthy)
    lake.upsert_agent_session(s_critical)

    tui.execute_key_action(
        "c",
        {
            "canonical_name": "storage_engine",
            "category": "architecture",
            "value": "SQLite WAL with bitemporal belief revisions",
            "pinned": True,
        },
    )

    rendered = tui.render_dashboard(width=140)
    assert "Context Palace" in rendered
    for pane in ("WINDOWS", "LAKE", "GRAPH", "AUDIT"):
        assert pane in rendered
    assert "LIVE" in rendered
    assert "agy-alpha" in rendered
    assert "gemini-2.5-pro" in rendered
    assert "25.00%" in rendered
    assert "HEALTHY" in rendered
    assert "claude-beta" in rendered
    assert "claude-sonnet-4" in rendered
    assert "86.00%" in rendered
    assert "COMPACT NOW" in rendered
    assert "storage_engine" in rendered
    assert "SQLite WAL with bitemporal belief revisions" in rendered
    assert "PINNED" in rendered
    assert "[c] Create" in rendered


def test_tui_key_action_create_and_read_history(temp_db: str) -> None:
    """Verify 'c' (Create) and 'r' (Read + Belief History) key actions."""
    comps = _load_components(temp_db)
    tui: MemoryPalaceTUI = comps["tui"]

    create_res = tui.execute_key_action(
        "c",
        {
            "canonical_name": "api_rate_limit",
            "category": "constraint",
            "value": "120 req/min per tenant",
            "confidence": 0.92,
            "timestamp": 1000.0,
        },
    )
    assert create_res["status"] == "ok"
    assert create_res["node"]["canonical_name"] == "api_rate_limit"

    read_res = tui.execute_key_action("r", {"node_id_or_name": "api_rate_limit"})
    assert read_res["status"] == "ok"
    assert read_res["node"]["active_value"] == "120 req/min per tenant"
    assert len(read_res["history"]) == 1
    assert read_res["history"][0]["value"] == "120 req/min per tenant"


def test_tui_key_action_update_preserves_bitemporal_revisions(temp_db: str) -> None:
    """Verify 'u' (Update) supersedes previous belief revision and updates active_value."""
    comps = _load_components(temp_db)
    tui: MemoryPalaceTUI = comps["tui"]

    tui.execute_key_action(
        "c",
        {
            "canonical_name": "vector_dim",
            "category": "config",
            "value": "32 dimensions",
            "timestamp": 1000.0,
        },
    )
    upd_res = tui.execute_key_action(
        "u",
        {
            "node_id_or_name": "vector_dim",
            "new_value": "64 dimensions L2-normalized",
            "confidence": 0.99,
            "timestamp": 2000.0,
        },
    )
    assert upd_res["status"] == "ok"
    assert upd_res["node"]["active_value"] == "64 dimensions L2-normalized"

    read_res = tui.execute_key_action("r", {"node_id_or_name": "vector_dim"})
    assert len(read_res["history"]) == 2
    old_rev, new_rev = read_res["history"][0], read_res["history"][1]
    assert old_rev["valid_to"] == 2000.0
    assert new_rev["valid_to"] is None
    assert new_rev["value"] == "64 dimensions L2-normalized"


def test_tui_key_action_delete_soft_and_hard(temp_db: str) -> None:
    """Verify 'd' (Delete) supports soft tombstoning and hard eviction."""
    comps = _load_components(temp_db)
    tui: MemoryPalaceTUI = comps["tui"]

    tui.execute_key_action(
        "c",
        {"canonical_name": "temp_feature_flag", "category": "config", "value": "enabled"},
    )
    soft_del = tui.execute_key_action(
        "d", {"node_id_or_name": "temp_feature_flag", "hard": False}
    )
    assert soft_del["deleted"] is True
    assert soft_del["hard"] is False

    # Soft-deleted concept should not appear in live dashboard table
    rendered = tui.render_dashboard()
    assert "(no active concepts)" in rendered
    assert "enabled" not in rendered

    # Re-create and hard-delete
    tui.execute_key_action(
        "c",
        {"canonical_name": "ephemeral_secret_ref", "category": "config", "value": "vault://k1"},
    )
    hard_del = tui.execute_key_action(
        "d", {"node_id_or_name": "ephemeral_secret_ref", "hard": True}
    )
    assert hard_del["deleted"] is True
    assert hard_del["hard"] is True


def test_tui_key_action_pin_and_inject_capsule(temp_db: str, tmp_path: Path) -> None:
    """Verify 'p' pins a concept and builds/injects an adaptive ContextCapsule into a sidecar file."""
    comps = _load_components(temp_db)
    lake = comps["lake"]
    tui: MemoryPalaceTUI = comps["tui"]

    lake.upsert_agent_session(
        AgentSession(
            agent_id="worker-delta",
            harness_type="antigravity",
            model_id="claude-sonnet-4",
            max_context_tokens=200_000,
            used_tokens=40_000,
        )
    )
    tui.execute_key_action(
        "c",
        {
            "canonical_name": "zero_dep_embeddings",
            "category": "architecture",
            "value": "Use 64-dim FNV-1a trigram hashing",
            "pinned": False,
        },
    )

    sidecar = tmp_path / "SIDECAR_MEMORY.md"
    res = tui.execute_key_action(
        "p",
        {
            "node_id_or_name": "zero_dep_embeddings",
            "pinned": True,
            "agent_id": "worker-delta",
            "query": "embeddings",
            "sidecar_path": str(sidecar),
        },
    )
    assert res["status"] == "ok"
    assert res["node"]["pinned"] is True
    assert res["capsule"] is not None
    assert res["capsule"]["target_agent_id"] == "worker-delta"
    assert sidecar.exists()
    assert "zero_dep_embeddings" in sidecar.read_text(encoding="utf-8")


def test_tui_key_action_refresh_model_windows(temp_db: str) -> None:
    """Verify 'm' discovers local agent sessions via registry and upserts them into the lake."""
    comps = _load_components(temp_db)
    lake = comps["lake"]

    class DummyRegistry:
        def discover_local_agents(
            self, search_dirs: Optional[List[str]] = None, limit: int = 10
        ) -> List[AgentSession]:
            return [
                AgentSession(
                    agent_id="discovered-ollama-1",
                    harness_type="ollama",
                    model_id="qwen2.5-coder:7b",
                    max_context_tokens=32_768,
                    used_tokens=8_192,
                )
            ]

    tui = MemoryPalaceTUI(lake=lake, retriever=comps["retriever"], registry=DummyRegistry())
    res = tui.execute_key_action("m", {"limit": 5})
    assert res["status"] == "ok"
    assert len(res["sessions"]) == 1
    assert res["sessions"][0]["agent_id"] == "discovered-ollama-1"
    assert res["sessions"][0]["max_context_tokens"] == 32_768


def test_tui_key_action_time_travel_snapshot(temp_db: str) -> None:
    """Verify 't' switches TUI into TIME-TRAVEL mode and reconstructs historical state at as_of."""
    comps = _load_components(temp_db)
    tui: MemoryPalaceTUI = comps["tui"]

    tui.execute_key_action(
        "c",
        {
            "canonical_name": "deployment_target",
            "category": "decision",
            "value": "Local Docker Compose",
            "timestamp": 1000.0,
        },
    )
    tui.execute_key_action(
        "u",
        {
            "node_id_or_name": "deployment_target",
            "new_value": "Railway Production Cluster",
            "timestamp": 2000.0,
        },
    )

    # Travel back to t=1500.0
    tt_res = tui.execute_key_action("t", {"as_of": 1500.0})
    assert tt_res["status"] == "ok"
    assert tt_res["as_of"] == 1500.0
    nodes_at_1500 = tt_res["snapshot"]["nodes"]
    assert len(nodes_at_1500) == 1
    val_at_1500 = nodes_at_1500[0].get("active_value") or nodes_at_1500[0].get("value")
    assert val_at_1500 == "Local Docker Compose"

    rendered_past = tui.render_dashboard()
    assert "TIME-TRAVEL as_of=1500.00" in rendered_past
    assert "Local Docker Compose" in rendered_past


def test_cli_crud_and_tui_once_workflow(temp_db: str, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI subcommands: crud create, list, update, read, pin, delete, and tui --once."""
    rc = main(
        [
            "--db",
            temp_db,
            "crud",
            "create",
            "--name",
            "consensus_policy",
            "--category",
            "architecture",
            "--value",
            "Provenance > Recency > Confidence",
        ]
    )
    assert rc == 0
    out_create = json.loads(capsys.readouterr().out)
    assert out_create["node"]["canonical_name"] == "consensus_policy"

    rc = main(["--db", temp_db, "crud", "list"])
    assert rc == 0
    out_list = json.loads(capsys.readouterr().out)
    assert out_list["count"] == 1

    rc = main(
        [
            "--db",
            temp_db,
            "crud",
            "update",
            "--name",
            "consensus_policy",
            "--value",
            "Strict Provenance Hierarchy v2",
            "--pinned",
        ]
    )
    assert rc == 0
    out_upd = json.loads(capsys.readouterr().out)
    assert out_upd["node"]["active_value"] == "Strict Provenance Hierarchy v2"

    rc = main(["--db", temp_db, "tui", "--once", "--search", "consensus"])
    assert rc == 0
    tui_out = capsys.readouterr().out
    assert "consensus_policy" in tui_out
    assert "Strict Provenance Hierarchy v2" in tui_out

    rc = main(["--db", temp_db, "crud", "delete", "--name", "consensus_policy"])
    assert rc == 0
    out_del = json.loads(capsys.readouterr().out)
    assert out_del["deleted"] is True


def test_cli_models_time_travel_and_inject(
    temp_db: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify CLI `models`, `time-travel`, and `inject` commands return exit code 0."""
    rc = main(["--db", temp_db, "models", "claude-sonnet-4", "gemini-2.5-pro"])
    assert rc == 0
    models_out = json.loads(capsys.readouterr().out)
    assert models_out["status"] == "ok"
    assert len(models_out["models"]) == 2

    main(
        [
            "--db",
            temp_db,
            "crud",
            "create",
            "--name",
            "cache_layer",
            "--category",
            "architecture",
            "--value",
            "SQLite FTS5 + 64d Cosine",
            "--pinned",
        ]
    )
    _ = capsys.readouterr()

    rc = main(["--db", temp_db, "time-travel", "--as-of", "9999999999.0"])
    assert rc == 0
    tt_out = json.loads(capsys.readouterr().out)
    assert tt_out["status"] == "ok"
    assert len(tt_out["snapshot"]["nodes"]) == 1

    capsule_file = tmp_path / "injected_capsule.md"
    rc = main(
        [
            "--db",
            temp_db,
            "inject",
            "--agent-id",
            "cli-agent-1",
            "--query",
            "cache",
            "--output",
            str(capsule_file),
        ]
    )
    assert rc == 0
    inj_out = json.loads(capsys.readouterr().out)
    assert inj_out["status"] == "ok"
    assert capsule_file.exists()
    assert "cache_layer" in capsule_file.read_text(encoding="utf-8")


def test_bare_cli_invocation_with_no_args(
    temp_db: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify bare `context-palace` (no subcommand) defaults to TUI without AttributeError."""
    rc = main(["--db", temp_db])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Context Palace" in out
    assert "[c] Create" in out
