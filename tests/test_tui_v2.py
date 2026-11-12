"""
Tests for the v2 TUI restyle: Rich view builders (`context_palace.tui_view`) and the
Textual app (`context_palace.tui_app`), driven with Textual's Pilot. Synthetic data only.
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import pytest

from context_palace.cli import _load_components, main
from context_palace.models import AgentSession
from context_palace.tui_view import (
    HOTKEYS,
    SPARK_CHARS,
    collect_graph_rows,
    sparkline,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from seed_demo_lake import seed  # noqa: E402

textual = pytest.importorskip("textual")
from context_palace.tui_app import ContextPalaceApp, PromptScreen  # noqa: E402


@pytest.fixture
def demo(tmp_path: Path) -> dict:
    db = str(tmp_path / "context_lake.db")
    seed(db).close()
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
        svg = app.export_screenshot()
        assert svg.startswith("<svg")
        assert "WINDOWS" in svg and "AUDIT" in svg
        assert "#e8efe6" in svg.lower()  # sage background token survives into the render

    _run_app(demo, scenario)
