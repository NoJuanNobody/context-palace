"""
Tests for the v2 TUI restyle: Rich view builders (`context_palace.tui_view`) and the
Textual app (`context_palace.tui_app`), driven with Textual's Pilot. Synthetic data only.
"""

from __future__ import annotations

import asyncio
import io
import json
import random
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import pytest

from context_palace.cli import _load_components, main
from context_palace.lake import ContextLake
from context_palace.models import AgentSession, ConflictResolutionEvent
from context_palace.tui_view import (
    HOTKEYS,
    SPARK_CHARS,
    collect_graph_rows,
    sparkline,
)

textual = pytest.importorskip("textual")
from context_palace.tui_app import ContextPalaceApp, PromptScreen  # noqa: E402


def _populate_test_lake(db_path: str, now: Optional[float] = None) -> ContextLake:
    now = now if now is not None else time.time()
    rng = random.Random(42)
    lake = ContextLake(db_path)

    sessions = [
        ("antigravity-3ad38af4", "antigravity", "gemini-4-argon", 1_000_000, 83_873, 142, 2_481_664),
        ("claude-code-dev", "claude_code", "claude-sonnet-4-5", 200_000, 168_400, 88, 891_204),
        ("codex-main", "codex", "gpt-4o", 128_000, 41_280, 51, 340_112),
        ("ollama-qwen-32b", "ollama", "qwen2.5-coder:32b", 32_768, 30_500, 44, 128_440),
    ]
    for agent_id, harness, model, mx, used, turn, off in sessions:
        lake.upsert_agent_session(
            AgentSession(
                agent_id=agent_id,
                harness_type=harness,
                model_id=model,
                max_context_tokens=mx,
                used_tokens=used,
                last_processed_turn=turn,
                last_byte_offset=off,
                updated_at=now,
            )
        )

    concepts = [
        ("federated_memory_topology", "architecture", "Hub-and-spoke SQLite WAL lake per workstation", 0.98, True, "operator_crud", "operator_cli"),
        ("database_engine", "architecture", "SQLite WAL with bitemporal belief revisions", 0.98, True, "verified_test", "antigravity-3ad38af4"),
        ("belief_revision_schema", "architecture", "valid_from / valid_to / superseded_by rows", 0.92, False, "agent_turn", "claude-code-dev"),
        ("dual_direction_retriever", "architecture", "Pull context_query + push context_inject_capsule", 0.91, False, "operator_crud", "operator_cli"),
        ("model_context_registry", "architecture", "Auto-detect model windows per harness", 0.90, False, "user_explicit", "codex-main"),
        ("token_compression_target", "constraint", "Capsules stay under 12% of remaining headroom", 0.99, True, "verified_test", "antigravity-3ad38af4"),
        ("pytest_48_passed_suite", "test", "48 tests green on main", 0.99, False, "verified_test", "codex-main"),
        ("agy_session_start", "hook", "Antigravity SessionStart hook injects capsule", 0.88, False, "agent_turn", "antigravity-3ad38af4"),
        ("auth_strategy", "architecture", "Use Ed25519 + WebAuthn passkeys", 0.97, True, "operator_crud", "operator_cli"),
        ("ed25519_key_rotation", "security", "Rotate signing keys every 30 days", 0.88, False, "agent_turn", "claude-code-dev"),
        ("mcp_server", "architecture", "stdio MCP server exposing context_* tools", 0.93, False, "operator_crud", "operator_cli"),
        ("context_query", "tool", "Pull: hybrid lexical + vector subgraph query", 0.90, False, "agent_turn", "codex-main"),
    ]
    start = now - 5.5 * 3600
    for i, (name, cat, val, conf, pinned, prov, agent) in enumerate(concepts):
        lake.create_concept(
            canonical_name=name,
            category=cat,
            value=val,
            confidence=conf,
            pinned=pinned,
            provenance_type=prov,
            source_agent_id=agent,
            timestamp=start + i * 300,
        )

    lake.update_concept("auth_strategy", "HS256 JWT, 1h TTL", confidence=0.70, provenance_type="agent_turn",
                        source_agent_id="antigravity-3ad38af4", timestamp=start + 3600)
    lake.update_concept("auth_strategy", "Ed25519 JWT, 15m TTL", confidence=0.86, provenance_type="verified_test",
                        source_agent_id="claude-code-dev", timestamp=start + 2 * 3600)
    lake.update_concept("auth_strategy", "Use Ed25519 + WebAuthn passkeys", confidence=0.97,
                        provenance_type="operator_crud", source_agent_id="operator_cli", timestamp=start + 4.5 * 3600)
    for _ in range(60):
        t = start + rng.betavariate(5, 2) * 5.4 * 3600
        name = concepts[rng.randrange(len(concepts))][0]
        if name in ("auth_strategy", "database_engine"):
            continue
        node = lake.get_concept(name)
        lake.update_concept(name, node.active_value, confidence=node.confidence,
                            provenance_type=lake.get_belief_history(name)[-1].provenance_type,
                            source_agent_id=node.source_agent_id, timestamp=t)

    edges = [
        ("federated_memory_topology", "database_engine", "depends_on"),
        ("database_engine", "belief_revision_schema", "configures"),
        ("federated_memory_topology", "dual_direction_retriever", "implements"),
        ("federated_memory_topology", "model_context_registry", "configures"),
        ("token_compression_target", "pytest_48_passed_suite", "verified_by"),
        ("token_compression_target", "agy_session_start", "hooked_into"),
        ("auth_strategy", "ed25519_key_rotation", "configures"),
        ("auth_strategy", "federated_memory_topology", "depends_on"),
        ("mcp_server", "context_query", "implements"),
    ]
    for src, dst, rel in edges:
        lake.add_edge(src, dst, rel, source_agent_id="operator_cli", timestamp=start)

    def last_two(name: str):
        hist = lake.get_belief_history(name)
        return hist[-1], hist[-2] if len(hist) > 1 else hist[-1]

    win, lose = last_two("auth_strategy")
    lake.record_conflict_event(ConflictResolutionEvent(
        event_id="cre_a91c0e2b44d1", node_id=lake.get_concept("auth_strategy").node_id,
        winning_revision_id=win.revision_id, losing_revision_id=lose.revision_id,
        winning_agent_id="operator_cli", losing_agent_id="antigravity-3ad38af4",
        winning_score=0.786, losing_score=0.635,
        reason="higher provenance authority (operator_crud=1.00 > agent_turn=0.70)",
        resolved_at=now - 1200,
    ))
    lake.update_concept("database_engine", "Postgres 16 with logical replication", confidence=0.62,
                        provenance_type="model_inference", source_agent_id="ollama-qwen-32b", timestamp=now - 2600)
    lake.update_concept("database_engine", "SQLite WAL with bitemporal belief revisions", confidence=0.98,
                        provenance_type="verified_test", source_agent_id="antigravity-3ad38af4", timestamp=now - 2400)
    win, lose = last_two("database_engine")
    lake.record_conflict_event(ConflictResolutionEvent(
        event_id="cre_b17f90c22188", node_id=lake.get_concept("database_engine").node_id,
        winning_revision_id=win.revision_id, losing_revision_id=lose.revision_id,
        winning_agent_id="antigravity-3ad38af4", losing_agent_id="ollama-qwen-32b",
        winning_score=0.819, losing_score=0.508,
        reason="higher confidence + verified_test provenance",
        resolved_at=now - 2400,
    ))
    return lake


@pytest.fixture
def demo(tmp_path: Path) -> dict:
    db = str(tmp_path / "context_lake.db")
    _populate_test_lake(db).close()
    return _load_components(db)


def _run_app(comps: dict, scenario: Callable[[ContextPalaceApp, Any], Awaitable[None]], size=(200, 52)) -> ContextPalaceApp:
    app = ContextPalaceApp(comps["tui"], refresh_seconds=0)

    async def _go() -> None:
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            await scenario(app, pilot)

    asyncio.run(_go())
    return app


def _static_text(app: ContextPalaceApp, widget_id: str) -> str:
    from rich.console import Console

    console = Console(width=200, record=True, file=io.StringIO(), color_system=None, force_terminal=False)
    console.print(app.pane_renderables[widget_id])
    return console.export_text()


# --------------------------------------------------------------------------- Rich render
def test_wide_render_has_four_panes_in_a_2x2_grid(demo: dict) -> None:
    out = demo["tui"].render_dashboard(width=200)
    lines = out.splitlines()
    assert any("WINDOWS" in ln and "LAKE" in ln for ln in lines), "WINDOWS and LAKE should share a row"
    assert any("GRAPH" in ln and "AUDIT" in ln for ln in lines), "GRAPH and AUDIT should share a row"
    for key, label in HOTKEYS:
        assert f"[{key}] {label}" in out
    assert "Context Palace" in out
    assert "LIVE" in out


def test_narrow_render_stacks_panes(demo: dict) -> None:
    out = demo["tui"].render_dashboard(width=120)
    lines = out.splitlines()
    assert not any("WINDOWS" in ln and "LAKE" in ln for ln in lines)
    order = [next(i for i, ln in enumerate(lines) if f"│ {p}" in ln) for p in ("WINDOWS", "LAKE", "GRAPH", "AUDIT")]
    assert order == sorted(order)


def test_panes_show_live_lake_data(demo: dict) -> None:
    tui = demo["tui"]
    lake = demo["lake"]
    out = tui.render_dashboard(width=200)
    # WINDOWS: real sessions, 2-decimal utilization, compaction pill
    assert "claude-code-dev" in out and "84.20%" in out and "COMPACT NOW" in out
    assert "( HEALTHY )" in out
    # LAKE: watermark straight from AgentSession
    assert "turn 142" in out and "2,481,664" in out
    # AUDIT: recorded ConflictResolutionEvent
    assert "cre_a91c0e2b44d1" in out
    # New data appears on the next render (no hardcoded mock rows)
    lake.create_concept(canonical_name="brand_new_concept", category="decision", value="ship v2", pinned=True)
    assert "brand_new_concept" in tui.render_dashboard(width=200)


def test_provenance_order_follows_consensus_py(demo: dict) -> None:
    from context_palace.consensus import PROVENANCE_WEIGHTS

    from rich.console import Console

    from context_palace.tui_view import build_lake_pane

    console = Console(width=400, record=True, file=io.StringIO(), color_system=None, force_terminal=False)
    console.print(build_lake_pane(demo["tui"].collect_dashboard_data()))
    flat = " ".join(console.export_text().split())
    ordered = sorted(PROVENANCE_WEIGHTS.items(), key=lambda kv: -kv[1])
    assert ordered[0][0] == "operator_crud"
    positions = [flat.index(f"{name} {w:.2f}") for name, w in ordered]
    assert positions == sorted(positions)


def test_graph_is_pinned_first_with_revision_provenance(demo: dict) -> None:
    data = demo["tui"].collect_dashboard_data()
    roots = [r for r in collect_graph_rows(data) if r.is_root]
    pinned_flags = [r.pinned for r in roots]
    assert pinned_flags == sorted(pinned_flags, reverse=True)
    auth = next(r for r in roots if r.name == "auth_strategy/")
    assert auth.provenance == "operator_crud"
    children = [r for r in collect_graph_rows(data) if r.relation == "configures" and r.name == "ed25519_key_rotation"]
    assert children and children[0].prefix.strip().startswith(("├─", "└─"))


def test_sparkline_buckets_activity() -> None:
    now = 10_000.0
    line = sparkline([now - 10, now - 10, now - 3000], end=now, buckets=8, window_s=8000)
    assert len(line.plain) == 8
    assert line.plain[-1] == SPARK_CHARS[-1]
    assert set(sparkline([], end=now).plain) == {SPARK_CHARS[0]}


def test_cli_once_width_flag(demo: dict, capsys: pytest.CaptureFixture[str]) -> None:
    db = demo["lake"].db_path
    assert main(["--db", db, "tui", "--once", "--width", "200"]) == 0
    out = capsys.readouterr().out
    assert any("WINDOWS" in ln and "LAKE" in ln for ln in out.splitlines())


# --------------------------------------------------------------------------- Textual app
def test_textual_app_mounts_panes_with_live_data(demo: dict) -> None:
    async def scenario(app: ContextPalaceApp, pilot: Any) -> None:
        assert "antigravity-3ad38af4" in _static_text(app, "windows")
        assert "CONCEPTS" in _static_text(app, "lake")
        assert "auth_strategy/" in _static_text(app, "graph")
        assert "ConflictResolutionEvent" in _static_text(app, "audit")
        status = _static_text(app, "statusbar")
        for key, label in HOTKEYS:
            assert f"[{key}] {label}" in status
        assert "textual" in status and "200×52" in status
        assert not app.query_one("#grid").has_class("stacked")

    _run_app(demo, scenario)


def test_textual_app_stacks_when_narrow(demo: dict) -> None:
    async def scenario(app: ContextPalaceApp, pilot: Any) -> None:
        assert app.query_one("#grid").has_class("stacked")

    _run_app(demo, scenario, size=(120, 40))


def test_textual_hotkeys_create_update_read_delete(demo: dict) -> None:
    lake = demo["lake"]

    async def scenario(app: ContextPalaceApp, pilot: Any) -> None:
        await pilot.press("c")
        await pilot.pause()
        assert isinstance(app.screen, PromptScreen)
        await pilot.press(*"cache_layer", "enter")
        await pilot.press("enter")  # keep default category
        await pilot.press(*"redis", "enter")
        await pilot.pause()
        assert lake.get_concept("cache_layer").active_value == "redis"
        assert "cache_layer" in _static_text(app, "graph")

        await pilot.press("u")
        await pilot.press(*"cache_layer", "enter", *"valkey", "enter")
        await pilot.pause()
        assert lake.get_concept("cache_layer").active_value == "valkey"

        await pilot.press("r")
        await pilot.press(*"cache_layer", "enter")
        await pilot.pause()
        audit = _static_text(app, "audit")
        assert "palace$ r cache_layer" in audit and "valkey" in audit

        await pilot.press("d")
        await pilot.press(*"cache_layer", "enter", "enter")  # blank = soft tombstone
        await pilot.pause()
        assert lake.get_concept("cache_layer") is None
        assert lake.get_concept("cache_layer", include_tombstoned=True) is not None

    _run_app(demo, scenario)


def test_textual_pin_inject_time_travel_models_and_quit(demo: dict) -> None:
    tui = demo["tui"]
    lake = demo["lake"]

    async def scenario(app: ContextPalaceApp, pilot: Any) -> None:
        await pilot.press("p")
        await pilot.pause()
        # target agent defaults to the worst offender (highest utilization)
        assert app.screen.query_one("#field-agent").value == "ollama-qwen-32b"
        await pilot.press(*"mcp_server", "enter", "enter", *"tools", "enter")
        await pilot.pause()
        assert lake.get_concept("mcp_server").pinned is True
        audit = _static_text(app, "audit")
        assert "context_inject_capsule  ollama-qwen-32b" in audit

        ts = str(time.time() - 3 * 3600)
        await pilot.press("t")
        await pilot.press(*ts, "enter")
        await pilot.pause()
        assert tui.as_of == pytest.approx(float(ts))
        assert "TIME-TRAVEL as_of=" in _static_text(app, "header")

        await pilot.press("t")
        await pilot.press("ctrl+u", "enter")  # clear -> LIVE
        await pilot.pause()
        assert tui.as_of is None

        await pilot.press("m")
        await pilot.pause()
        assert "palace$ m" in _static_text(app, "audit")

        await pilot.press("escape")
        await pilot.press("q")
        await pilot.pause()

    app = _run_app(demo, scenario)
    assert app.return_code in (0, None)


def test_textual_prompt_escape_cancels(demo: dict) -> None:
    async def scenario(app: ContextPalaceApp, pilot: Any) -> None:
        await pilot.press("d")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, PromptScreen)
        assert "Cancelled" in _static_text(app, "audit")

    _run_app(demo, scenario)


def test_textual_screenshot_svg(demo: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)

    async def scenario(app: ContextPalaceApp, pilot: Any) -> None:
        svg_dark = app.export_screenshot()
        assert svg_dark.startswith("<svg")
        assert "WINDOWS" in svg_dark and "AUDIT" in svg_dark
        assert "#0d1210" in svg_dark.lower()  # default dark mode obsidian background token

        # Toggle to light mode via 'k' hotkey and verify sage background token
        await pilot.press("k")
        await pilot.pause()
        svg_light = app.export_screenshot()
        assert "#e8efe6" in svg_light.lower()

    _run_app(demo, scenario)


def test_textual_clickable_crud_modals_and_actions(demo: dict) -> None:
    from context_palace.tui_app import AgentCrudScreen, ConceptCrudScreen, CrudHubScreen

    lake = demo["lake"]

    async def scenario(app: ContextPalaceApp, pilot: Any) -> None:
        # 1. Click a concept in GRAPH -> opens ConceptCrudScreen -> update belief value & add edge
        app.action_open_concept_crud("auth_strategy")
        await pilot.pause()
        assert isinstance(app.screen, ConceptCrudScreen)
        val_input = app.screen.query_one("#crud-value")
        val_input.value = "Zero-trust Ed25519 + mTLS"
        edge_tgt = app.screen.query_one("#crud-edge-target")
        edge_tgt.value = "mcp_server"
        app.screen.action_save()
        await pilot.pause()
        assert lake.get_concept("auth_strategy").active_value == "Zero-trust Ed25519 + mTLS"

        # 2. Quick toggle pin & quick delete via clickable actions
        orig_pin = lake.get_concept("mcp_server").pinned
        app.action_quick_toggle_pin("mcp_server")
        await pilot.pause()
        assert lake.get_concept("mcp_server").pinned is (not orig_pin)

        # 3. Click an agent row -> opens AgentCrudScreen -> update used tokens
        app.action_manage_agent("claude-code-dev")
        await pilot.pause()
        assert isinstance(app.screen, AgentCrudScreen)
        app.screen.query_one("#agent-used").value = "95000"
        app.screen.action_save()
        await pilot.pause()
        assert lake.get_agent_session("claude-code-dev").used_tokens == 95000

        # 4. Open CRUD Manager Hub -> soft delete a concept and restore it
        app.action_open_crud_hub()
        await pilot.pause()
        assert isinstance(app.screen, CrudHubScreen)
        app.screen.action_soft_del("mcp_server")
        assert lake.get_concept("mcp_server") is None
        app.screen.action_restore_concept("mcp_server")
        assert lake.get_concept("mcp_server") is not None
        app.screen.action_close()
        await pilot.pause()

    _run_app(demo, scenario)


def test_textual_ask_hotkey_runs_conversation_turn_into_audit(demo: dict) -> None:
    tui = demo["tui"]
    lake = demo["lake"]

    async def scenario(app: ContextPalaceApp, pilot: Any) -> None:
        await pilot.press("a")
        await pilot.pause()
        assert isinstance(app.screen, PromptScreen)
        await pilot.press(*"What is the auth_strategy?", "enter")
        await pilot.pause()
        assert len(tui.chat_history) == 1
        audit = _static_text(app, "audit")
        assert "You:    What is the auth_strategy?" in audit
        assert "Palace: " in audit

        # Natural-language CRUD through the same hotkey
        await pilot.press("a")
        await pilot.press(*"Remember api_port is 9000", "enter")
        await pilot.pause()
        node = lake.get_concept("api_port")
        assert node is not None and node.active_value == "9000"
        assert len(tui.chat_history) == 2

    _run_app(demo, scenario)


def test_textual_list_all_hotkey_opens_untruncated_inventory(demo: dict) -> None:
    from context_palace.tui_app import InventoryScreen

    tui = demo["tui"]

    async def scenario(app: ContextPalaceApp, pilot: Any) -> None:
        await pilot.press("l")
        await pilot.pause()
        assert isinstance(app.screen, InventoryScreen)
        assert "Lake Inventory" in app.screen.body
        assert "Listed all" in tui.status_message
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, InventoryScreen)
        assert "palace$ l" in _static_text(app, "audit")

    _run_app(demo, scenario)


def test_status_bar_lists_chat_and_list_all_keys(demo: dict) -> None:
    out = demo["tui"].render_dashboard(width=200)
    for key in ("c", "r", "u", "d", "l", "a", "p", "m", "t", "q"):
        assert f"[{key}]" in out


def test_topic_and_concept_summaries_in_all_four_panes(demo: dict) -> None:
    from context_palace.summarizer import summarize_3_to_5_words, summarize_concept

    s_topic = summarize_3_to_5_words("i want the functionality of terminal for one click install")
    assert 3 <= len(s_topic.split()) <= 5
    assert "Terminal" in s_topic and "Install" in s_topic

    c_info = summarize_concept("auth_strategy", "Use Ed25519 + WebAuthn passkeys", category="architecture")
    assert "Ed25519" in c_info["short_topic"]
    assert "WebAuthn" in c_info["plain_english"]

    out = demo["tui"].render_dashboard(width=200)
    assert "AGENT / TOPIC" in out
    assert "LAKE TOPIC" in out
    assert "Active Topic Summaries" in out
    assert "Ed25519" in out


def test_visual_graph_shape_cards_sort_order_and_nl_agent(demo: dict) -> None:
    from context_palace.tui_app import ConceptCrudScreen

    tui = demo["tui"]
    lake = demo["lake"]

    out = tui.render_dashboard(width=200)
    # 1. Visual Node Shape Cards & View Order bar in GRAPH
    assert "VIEW ORDER:" in out
    assert "1. ★ Pinned & Recent" in out
    assert "NODE SHAPES:  ▲ Architecture" in out
    # 2. Selected Node Explanation in AUDIT
    assert "SELECTED NODE EXPLANATION" in out
    assert "What this represents:" in out
    assert "Why it matters:" in out

    # 3. Interactive Textual actions: select node, change sort order, and update via Natural-Language Terminal Agent
    async def scenario(app: ContextPalaceApp, pilot: Any) -> None:
        # Select a node -> updates Selected Node Explanation in AUDIT without opening a modal form
        app.action_select_node("auth_strategy")
        await pilot.pause()
        assert tui.selected_node_id == "auth_strategy"
        audit_txt = _static_text(app, "audit")
        assert "SELECTED NODE EXPLANATION" in audit_txt and "(auth_strategy)" in audit_txt
        assert "What this represents:" in audit_txt

        # Change viewing order via set_sort_order
        app.action_set_sort_order("category")
        await pilot.pause()
        assert tui.graph_sort_order == "category"

        # Update node via Natural-Language Terminal Agent inside ConceptCrudScreen
        app.action_open_concept_crud("auth_strategy")
        await pilot.pause()
        assert isinstance(app.screen, ConceptCrudScreen)
        nl_in = app.screen.query_one("#crud-nl-input")
        nl_in.value = "Change auth_strategy to Use Passkeys + Hardware YubiKey, set confidence to 99%, and link to mcp_server"
        app.screen.action_apply_nl()
        await pilot.pause()

        updated = lake.get_concept("auth_strategy")
        assert "Passkeys + Hardware YubiKey" in updated.active_value
        assert updated.confidence == pytest.approx(0.99)
        edges = lake.list_edges(node_id=updated.node_id, active_only=True)
        mcp_node = lake.get_concept("mcp_server")
        assert any(e.target_id == mcp_node.node_id or e.source_id == mcp_node.node_id for e in edges)

    _run_app(demo, scenario)


