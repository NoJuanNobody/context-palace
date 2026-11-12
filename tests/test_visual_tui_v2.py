"""
Comprehensive Unit & Integration Tests for FigCraft Visual TUI v2.0, CLI list-all & chat,
and MCP tools (context_list_all, context_chat).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from context_palace.cli import _load_components, main
from context_palace.lake import ContextLake
from context_palace.mcp_server import ContextPalaceMCPServer
from context_palace.models import AgentSession, ConceptNode, RelationEdge
from context_palace.tui import MemoryPalaceTUI, render_unicode_bar


@pytest.fixture
def temp_db(tmp_path: Path) -> str:
    return str(tmp_path / "test_palace_v2.db")


# ==============================================================================
# 1. Unicode Progress Bar Helper Tests
# ==============================================================================


def test_render_unicode_bar_color_coding_and_widths() -> None:
    """Verify render_unicode_bar generates correct █/░ ratios and color codes."""
    # Green tier (< 60%)
    bar_low = render_unicode_bar(0.25, width=12)
    assert "[green]" in bar_low
    assert "[/green]" in bar_low
    assert "███░░░░░░░░░" in bar_low

    # Halfway (50% -> green)
    bar_half = render_unicode_bar(0.50, width=12)
    assert "[green]" in bar_half
    assert "██████░░░░░░" in bar_half

    # Yellow tier (60% <= ratio < 85%)
    bar_mid = render_unicode_bar(0.70, width=10)
    assert "[yellow]" in bar_mid
    assert "███████░░░" in bar_mid

    # Red tier (>= 85%)
    bar_high = render_unicode_bar(0.90, width=10)
    assert "[red]" in bar_high
    assert "█████████░" in bar_high

    # Clamping & edge cases
    bar_zero = render_unicode_bar(-0.5, width=8, colored=False)
    assert bar_zero == "░" * 8

    bar_full = render_unicode_bar(1.5, width=8, colored=False)
    assert bar_full == "█" * 8


# ==============================================================================
# 2. FigCraft Visual TUI v2.0 3-Pane Layout Tests
# ==============================================================================


def test_visual_tui_v2_three_pane_layout(temp_db: str) -> None:
    """Verify render_to_text displays 3 panes, Unicode bars, badges, graph cards, and inspector."""
    lake = ContextLake(temp_db)
    lake.upsert_agent_session(
        AgentSession(
            agent_id="agent-claude",
            harness_type="claude_code",
            model_id="claude-3-7-sonnet",
            max_context_tokens=200_000,
            used_tokens=50_000,
            last_byte_offset=12400,
            last_processed_turn=14,
        )
    )
    lake.upsert_agent_session(
        AgentSession(
            agent_id="agent-gemini",
            harness_type="antigravity",
            model_id="gemini-2.5-pro",
            max_context_tokens=1_000_000,
            used_tokens=880_000,  # 88% -> COMPACT!
            last_byte_offset=580000,
            last_processed_turn=62,
        )
    )

    n1 = lake.create_concept(
        canonical_name="federated_memory_topology",
        category="architecture",
        value="CRDT gossip mesh across local agent worktrees",
        confidence=0.98,
        pinned=True,
    )
    n2 = lake.create_concept(
        canonical_name="token_compression_target",
        category="constraint",
        value=">=95% reduction via Context Capsules",
        confidence=0.92,
        pinned=False,
    )
    lake.add_edge(
        source_id=n1.node_id,
        target_id=n2.node_id,
        relation_type="constrains",
        weight=0.95,
        source_agent_id="system",
    )

    tui = MemoryPalaceTUI(lake)
    rendered = tui.render_to_text(width=140)

    # Header (four-pane v2 design, PR #2)
    assert "Context Palace" in rendered
    assert "LIVE" in rendered

    # WINDOWS + LAKE: sessions, tier distribution, watermarks
    assert "WINDOWS" in rendered
    assert "agent-claude" in rendered
    assert "claude-3-7-sonnet" in rendered
    assert "( HEALTHY )" in rendered
    assert "agent-gemini" in rendered
    assert "gemini-2.5-pro" in rendered
    assert "COMPACT NOW" in rendered
    assert "LAKE" in rendered
    assert "TIERS" in rendered
    assert "hot 2" in rendered  # pinned topology + conf 0.92 constraint
    assert "warm 0" in rendered
    assert "12,400" in rendered and "turn 14" in rendered  # extraction watermark

    # GRAPH: pinned-first tree with edges
    assert "GRAPH" in rendered
    assert "federated_memory_topology" in rendered
    assert "CRDT gossip mesh across local agent worktrees" in rendered
    assert "token_compression_target" in rendered
    assert "PINNED" in rendered
    assert "constrains" in rendered

    # AUDIT: conversational engine lives here; status bar carries every hotkey
    assert "AUDIT" in rendered
    assert "[a] Ask the palace" in rendered
    assert "[c] Create" in rendered
    assert "[l] List All" in rendered
    assert "[a] Ask" in rendered


def test_visual_tui_v2_time_travel_scrubber(temp_db: str) -> None:
    """Verify Bitemporal Time-Travel Scrubber changes state when as_of is active."""
    lake = ContextLake(temp_db)
    tui = MemoryPalaceTUI(lake)

    # Initial LIVE mode
    live_out = tui.render_to_text()
    assert "LIVE" in live_out
    assert "TIME-TRAVEL" not in live_out

    # Activate time travel
    res = tui.execute_key_action("t", {"as_of": 1700000000.0})
    assert res["status"] == "ok"
    assert res["as_of"] == 1700000000.0

    past_out = tui.render_to_text()
    assert "TIME-TRAVEL as_of=1700000000.00" in past_out


# ==============================================================================
# 3. Interactive TUI Commands [l] (list-all) and [a] (ask/chat)
# ==============================================================================


def test_visual_tui_v2_commands_list_all_and_chat(temp_db: str) -> None:
    """Verify 'l' (list-all) and 'a' (ask/chat) handle_command actions in TUI."""
    lake = ContextLake(temp_db)
    lake.upsert_agent_session(
        AgentSession(
            agent_id="test-agent",
            harness_type="test",
            model_id="gemini-flash",
            max_context_tokens=1000,
            used_tokens=200,
        )
    )
    lake.create_concept(
        canonical_name="cache_port",
        category="config",
        value="8443",
    )

    tui = MemoryPalaceTUI(lake)

    # Test [l] / list-all
    l_res = tui.handle_command("l")
    assert l_res["status"] == "ok"
    assert l_res["action"] == "list-all"
    assert l_res["nodes"] == 1
    assert l_res["sessions"] == 1
    assert "Listed all 1 nodes, 0 edges, 1 sessions" in tui.status_message

    # Test [a] / ask conversational query
    a_res = tui.handle_command("a What is the cache_port?")
    assert a_res["status"] == "ok"
    assert a_res["action"] == "chat"
    assert "8443" in a_res["answer"]
    assert "cache_port" in a_res["cited_nodes"]
    assert len(tui.chat_history) == 1
    assert tui.chat_history[0]["user"] == "What is the cache_port?"

    # Right pane now reflects the conversation turn
    rendered = tui.render_to_text()
    assert "You:    What is the cache_port?" in rendered
    assert "Palace: Found 1 relevant belief(s)" in rendered

    # Test Natural Language CRUD via chat
    crud_res = tui.handle_command("chat Remember api_port is 9000")
    assert crud_res["status"] == "ok"
    assert crud_res["intent"] == "create"
    assert "api_port" in crud_res["cited_nodes"]

    # Verify concept exists in lake
    created_node = lake.get_concept("api_port")
    assert created_node is not None
    assert created_node.active_value == "9000"


# ==============================================================================
# 4. CLI Subcommands: list-all (and ls) Formats & Chat
# ==============================================================================


def test_cli_list_all_json_markdown_and_rich(
    temp_db: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify CLI list-all outputs complete untruncated inventory across 5 tables in all formats."""
    lake = ContextLake(temp_db)
    lake.upsert_agent_session(
        AgentSession(
            agent_id="agent-omega",
            harness_type="antigravity",
            model_id="gemini-2.5-flash",
            max_context_tokens=1_048_576,
            used_tokens=42_000,
            last_byte_offset=1024,
            last_processed_turn=5,
        )
    )
    lake.create_concept(
        canonical_name="memory_store",
        category="architecture",
        value="SQLite WAL mode with bitemporal valid_to tracking",
        confidence=0.99,
        pinned=True,
    )

    # 1. Format: JSON
    rc = main(["--db", temp_db, "list-all", "--format", "json"])
    assert rc == 0
    raw_json = capsys.readouterr().out
    data = json.loads(raw_json)
    assert "memory_nodes" in data
    assert "memory_edges" in data
    assert "agent_sessions" in data
    assert "extraction_watermarks" in data
    assert "model_registry" in data
    assert data["counts"]["memory_nodes"] == 1
    assert data["counts"]["agent_sessions"] == 1
    assert data["memory_nodes"][0]["canonical_name"] == "memory_store"
    assert data["agent_sessions"][0]["agent_id"] == "agent-omega"

    # 2. Format: Markdown
    rc = main(["--db", temp_db, "list-all", "--format", "markdown"])
    assert rc == 0
    md_out = capsys.readouterr().out
    assert "# Context Palace Complete Lake Inventory" in md_out
    assert "## 1. Concept Nodes (`memory_nodes`)" in md_out
    assert "memory_store" in md_out
    assert "agent-omega" in md_out
    assert "## 4. Extraction Watermarks (`extraction_watermarks`)" in md_out

    # 3. Format: Rich / ASCII (via alias 'ls')
    rc = main(["--db", temp_db, "ls", "--format", "rich"])
    assert rc == 0
    rich_out = capsys.readouterr().out
    assert "CONTEXT PALACE — FULL UNTRUNCATED LAKE INVENTORY (5 TABLES)" in rich_out
    assert "--- [Table 1: Memory Nodes / Concepts] ---" in rich_out
    assert "memory_store" in rich_out
    assert "[PINNED]" in rich_out
    assert "--- [Table 3: Agent Sessions] ---" in rich_out
    assert "agent-omega" in rich_out


def test_cli_chat_single_shot_and_json(
    temp_db: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify CLI chat single-shot conversational query and JSON mode."""
    lake = ContextLake(temp_db)
    lake.create_concept(
        canonical_name="auth_provider",
        category="security",
        value="Ed25519 sealed envelopes TOFU",
    )

    # Plain single shot
    rc = main(["--db", temp_db, "chat", "What is auth_provider?"])
    assert rc == 0
    plain_out = capsys.readouterr().out
    assert "Palace:" in plain_out
    assert "Ed25519 sealed envelopes TOFU" in plain_out
    assert "Cited concepts: auth_provider" in plain_out

    # JSON single shot
    rc = main(["--db", temp_db, "converse", "What is auth_provider?", "--json"])
    assert rc == 0
    json_out = json.loads(capsys.readouterr().out)
    assert json_out["status"] == "ok"
    assert "auth_provider" in json_out["cited_nodes"]
    assert "Ed25519" in json_out["answer"]


# ==============================================================================
# 5. MCP Tools: context_list_all and context_chat
# ==============================================================================


def test_mcp_context_list_all_and_context_chat(temp_db: str) -> None:
    """Verify MCP tools context_list_all and context_chat schemas and execution."""
    lake = ContextLake(temp_db)
    lake.upsert_agent_session(
        AgentSession(
            agent_id="mcp-agent",
            harness_type="mcp",
            model_id="claude-opus-4",
            max_context_tokens=200_000,
            used_tokens=10_000,
        )
    )
    lake.create_concept(
        canonical_name="ipc_protocol",
        category="architecture",
        value="Unix domain sockets with JSON-RPC 2.0",
        confidence=0.96,
        pinned=True,
    )

    server = ContextPalaceMCPServer(lake)

    # Verify tool definitions
    defs = server.get_tool_definitions()
    tool_names = [d["name"] for d in defs]
    assert "context_list_all" in tool_names
    assert "context_chat" in tool_names

    # Test context_list_all via _dispatch_tool
    inv_res = server._dispatch_tool("context_list_all", {})
    assert inv_res["status"] == "ok"
    assert "memory_nodes" in inv_res
    assert "memory_edges" in inv_res
    assert "agent_sessions" in inv_res
    assert "extraction_watermarks" in inv_res
    assert "model_registry" in inv_res
    assert inv_res["counts"]["memory_nodes"] == 1
    assert inv_res["counts"]["agent_sessions"] == 1
    assert inv_res["memory_nodes"][0]["canonical_name"] == "ipc_protocol"

    # Test context_chat query via call_tool
    chat_res = server.call_tool("context_chat", {"message": "What is ipc_protocol?"})
    assert chat_res["status"] == "ok"
    assert "ipc_protocol" in chat_res["cited_nodes"]
    assert "Unix domain sockets" in chat_res["answer"]
    assert chat_res["intent"] == "query"

    # Test context_chat natural language CRUD
    crud_chat_res = server.call_tool(
        "context_chat", {"message": "Remember debug_mode is true"}
    )
    assert crud_chat_res["status"] == "ok"
    assert crud_chat_res["intent"] == "create"
    assert "debug_mode" in crud_chat_res["cited_nodes"]

    node = lake.get_concept("debug_mode")
    assert node is not None
    assert node.active_value == "true"
