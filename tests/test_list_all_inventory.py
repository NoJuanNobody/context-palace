"""
Unit tests for Context Palace LakeInventoryExporter, list_everything, and render_list_all_report.
Covers:
  - Empty lake inventory export
  - Populated lake with active + superseded nodes, edges, sessions, watermarks, and models
  - Filtering by include_superseded=False, agent_filter, and memory_type_filter
  - Rendering in "json", "markdown", and "rich" formats ensuring zero truncation of long content.
"""

from __future__ import annotations

import json
import pytest

from context_palace.lake import ContextLake
from context_palace.inventory import (
    LakeInventoryExporter,
    list_everything,
    render_list_all_report,
)
from context_palace.models import AgentSession, ModelWindowSpec
from context_palace.model_registry import ModelContextRegistry


def test_empty_lake_inventory() -> None:
    """Verify empty lake export returns clean zeroed summary and empty collections."""
    lake = ContextLake(":memory:")
    exporter = LakeInventoryExporter(lake)
    inv = exporter.export_inventory()

    assert "summary" in inv
    assert "nodes" in inv
    assert "edges" in inv
    assert "sessions" in inv
    assert "watermarks" in inv
    assert "models" in inv

    summary = inv["summary"]
    assert summary["active_nodes"] == 0
    assert summary["superseded_nodes"] == 0
    assert summary["total_nodes"] == 0
    assert summary["total_edges"] == 0
    assert summary["total_sessions"] == 0
    assert summary["total_watermarks"] == 0
    assert summary["registered_models"] == 0
    assert summary["token_estimates"] == 0

    assert inv["nodes"] == []
    assert inv["edges"] == []
    assert inv["sessions"] == []
    assert inv["watermarks"] == []
    assert inv["models"] == []

    # Render formats on empty lake
    json_out = render_list_all_report(inv, format="json")
    parsed = json.loads(json_out)
    assert parsed["summary"]["total_nodes"] == 0

    md_out = render_list_all_report(inv, format="markdown")
    assert "# Context Palace — Lake Inventory Report" in md_out
    assert "*(no memory nodes)*" in md_out
    assert "*(no edges)*" in md_out
    assert "*(no agent sessions)*" in md_out
    assert "*(no extraction watermarks)*" in md_out
    assert "*(no models registered)*" in md_out

    rich_out = render_list_all_report(inv, format="rich")
    assert "Lake Inventory Summary" in rich_out
    assert "Memory Nodes" in rich_out


def test_populated_lake_inventory_direct_tables() -> None:
    """Verify populated lake with memory_nodes, memory_edges, sessions, watermarks, and model_registry."""
    lake = ContextLake(":memory:")
    exporter = LakeInventoryExporter(lake)

    # 1. Insert memory nodes (2 active, 1 superseded)
    lake._conn.execute(
        """
        INSERT INTO memory_nodes (node_id, memory_type, tier, source_agent, confidence, valid_from, valid_until, superseded_by, content)
        VALUES
          ('node-arch-1', 'architecture', 'standard', 'agent-flash', 0.95, 100.0, NULL, NULL, 'System uses event-driven CQRS pattern'),
          ('node-db-v1', 'config', 'standard', 'agent-flash', 0.80, 50.0, 150.0, 'node-db-v2', 'Database host is db.internal:5432'),
          ('node-db-v2', 'config', 'pinned', 'agent-claude', 0.99, 150.0, NULL, NULL, 'Database moved to SQLite WAL memory palace')
        """
    )

    # 2. Insert memory edge
    lake._conn.execute(
        """
        INSERT INTO memory_edges (edge_id, source_id, target_id, relation_type, weight, source_agent, valid_from, valid_until)
        VALUES ('edge-001', 'node-arch-1', 'node-db-v2', 'depends_on', 1.0, 'agent-flash', 150.0, NULL)
        """
    )

    # 3. Insert agent session
    session = AgentSession(
        agent_id="agent-flash",
        harness_type="antigravity",
        model_id="gemini-2.5-flash",
        max_context_tokens=1_000_000,
        used_tokens=42_000,
        transcript_path="/tmp/transcript.jsonl",
        last_processed_turn=10,
        last_byte_offset=8192,
        updated_at=1700000000.0,
    )
    lake.upsert_agent_session(session)

    # 4. Insert extraction watermark
    lake._conn.execute(
        """
        INSERT INTO extraction_watermarks (agent_id, transcript_path, last_turn, last_offset, updated_at)
        VALUES ('agent-flash', '/tmp/transcript.jsonl', 10, 8192, 1700000000.0)
        """
    )

    # 5. Insert model_registry entry & custom registry
    lake._conn.execute(
        """
        INSERT INTO model_registry (model_id, canonical_name, provider, max_context_tokens, max_output_tokens, source)
        VALUES ('custom-fast', 'Custom Fast 64K', 'custom', 65536, 8192, 'db')
        """
    )
    lake._conn.commit()

    custom_reg = ModelContextRegistry(
        custom_specs={
            "agent-spec-1": ModelWindowSpec(
                model_id="agent-spec-1",
                canonical_name="Agent Spec 128K",
                provider="anthropic",
                max_context_tokens=131072,
            )
        }
    )
    exporter = LakeInventoryExporter(lake, registry=custom_reg)
    inv = exporter.export_inventory(include_superseded=True)

    summary = inv["summary"]
    assert summary["total_nodes"] == 3
    assert summary["active_nodes"] == 2
    assert summary["superseded_nodes"] == 1
    assert summary["total_edges"] == 1
    assert summary["total_sessions"] == 1
    assert summary["total_watermarks"] == 1
    assert summary["registered_models"] >= 2
    assert summary["token_estimates"] > 0

    # Verify order of nodes: valid_from DESC, node_id ASC
    # Node timestamps: node-db-v2 (150.0), node-arch-1 (100.0), node-db-v1 (50.0)
    nodes = inv["nodes"]
    assert len(nodes) == 3
    assert nodes[0]["node_id"] == "node-db-v2"
    assert nodes[1]["node_id"] == "node-arch-1"
    assert nodes[2]["node_id"] == "node-db-v1"

    # Verify edge details
    edges = inv["edges"]
    assert len(edges) == 1
    assert edges[0]["edge_id"] == "edge-001"
    assert edges[0]["relation_type"] == "depends_on"

    # Verify session details
    sessions = inv["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["agent_id"] == "agent-flash"
    assert sessions[0]["used_tokens"] == 42_000

    # Verify watermark details
    watermarks = inv["watermarks"]
    assert len(watermarks) == 1
    assert watermarks[0]["agent_id"] == "agent-flash"
    assert watermarks[0]["last_turn"] == 10

    # Verify models combined from DB and registry
    model_ids = {m["model_id"].lower() for m in inv["models"]}
    assert "custom-fast" in model_ids
    assert "agent-spec-1" in model_ids


def test_populated_lake_via_concepts_and_belief_revisions() -> None:
    """Verify inventory export falls back seamlessly to concept_nodes + belief_revisions."""
    lake = ContextLake(":memory:")

    # Create concept node
    lake.create_concept(
        canonical_name="CacheStrategy",
        category="architecture",
        value="Initial in-memory LRU cache",
        source_agent_id="agent-01",
        pinned=True,
        timestamp=100.0,
        node_id="concept-cache",
    )
    # Update concept to create a superseded revision
    lake.update_concept(
        node_id_or_name="concept-cache",
        new_value="Distributed Redis + SQLite Cache",
        source_agent_id="agent-02",
        timestamp=200.0,
    )
    lake.add_edge("concept-cache", "concept-cache", "self_loop", timestamp=150.0)

    inv = list_everything(lake, include_superseded=True)

    assert inv["summary"]["total_nodes"] == 2
    assert inv["summary"]["active_nodes"] == 1
    assert inv["summary"]["superseded_nodes"] == 1
    assert inv["summary"]["total_edges"] == 1

    nodes = inv["nodes"]
    assert len(nodes) == 2
    # Newest revision should be first
    assert nodes[0]["valid_from"] == 200.0
    assert nodes[0]["content"] == "Distributed Redis + SQLite Cache"
    assert nodes[0]["superseded_by"] is None

    assert nodes[1]["valid_from"] == 100.0
    assert nodes[1]["content"] == "Initial in-memory LRU cache"
    assert nodes[1]["superseded_by"] is not None


def test_inventory_filtering() -> None:
    """Test filtering by include_superseded=False, agent_filter, and memory_type_filter."""
    lake = ContextLake(":memory:")
    exporter = LakeInventoryExporter(lake)

    lake._conn.execute(
        """
        INSERT INTO memory_nodes (node_id, memory_type, tier, source_agent, confidence, valid_from, valid_until, superseded_by, content)
        VALUES
          ('node-1', 'architecture', 'standard', 'agent-alpha', 0.95, 100.0, NULL, NULL, 'Arch Alpha Active'),
          ('node-2', 'architecture', 'standard', 'agent-alpha', 0.90, 50.0, 100.0, 'node-1', 'Arch Alpha Superseded'),
          ('node-3', 'decision', 'standard', 'agent-beta', 0.92, 120.0, NULL, NULL, 'Decision Beta Active'),
          ('node-4', 'config', 'pinned', 'agent-alpha', 0.99, 130.0, NULL, NULL, 'Config Alpha Active')
        """
    )
    lake._conn.commit()

    # 1. include_superseded=False
    inv_active = exporter.export_inventory(include_superseded=False)
    assert inv_active["summary"]["total_nodes"] == 3
    assert inv_active["summary"]["active_nodes"] == 3
    assert inv_active["summary"]["superseded_nodes"] == 0
    assert all(n["superseded_by"] is None and n["valid_until"] is None for n in inv_active["nodes"])

    # 2. agent_filter="agent-alpha" (with superseded)
    inv_alpha = exporter.export_inventory(include_superseded=True, agent_filter="agent-alpha")
    assert inv_alpha["summary"]["total_nodes"] == 3
    assert all(n["source_agent"] == "agent-alpha" for n in inv_alpha["nodes"])

    # 3. agent_filter="agent-alpha" (active only)
    inv_alpha_active = exporter.export_inventory(include_superseded=False, agent_filter="agent-alpha")
    assert inv_alpha_active["summary"]["total_nodes"] == 2
    assert {n["node_id"] for n in inv_alpha_active["nodes"]} == {"node-1", "node-4"}

    # 4. memory_type_filter="decision"
    inv_decision = exporter.export_inventory(include_superseded=True, memory_type_filter="decision")
    assert inv_decision["summary"]["total_nodes"] == 1
    assert inv_decision["nodes"][0]["node_id"] == "node-3"

    # 5. Combined filter: agent_filter="agent-alpha" and memory_type_filter="config"
    inv_combined = exporter.export_inventory(
        include_superseded=False,
        agent_filter="agent-alpha",
        memory_type_filter="config",
    )
    assert len(inv_combined["nodes"]) == 1
    assert inv_combined["nodes"][0]["node_id"] == "node-4"


def test_zero_truncation_rendering() -> None:
    """Verify that multi-line long node contents are rendered with zero truncation across formats."""
    lake = ContextLake(":memory:")
    exporter = LakeInventoryExporter(lake)

    long_chunk = (
        "This is paragraph line of critical context information with high entropy and tokens. "
        "Every single byte must be preserved without truncation, ellipsis, or line slicing.\n"
    )
    sentinel_start = ">>>START_ZERO_TRUNCATION_TEST_PAYLOAD<<<"
    sentinel_end = ">>>END_ZERO_TRUNCATION_TEST_PAYLOAD<<<"
    large_untruncated_content = f"{sentinel_start}\n" + (long_chunk * 40) + f"{sentinel_end}"

    lake._conn.execute(
        """
        INSERT INTO memory_nodes (node_id, memory_type, tier, source_agent, confidence, valid_from, valid_until, superseded_by, content)
        VALUES ('node-massive', 'spec', 'pinned', 'agent-pro', 0.99, 500.0, NULL, NULL, ?)
        """,
        (large_untruncated_content,),
    )
    lake._conn.commit()

    inv = exporter.export_inventory()
    assert inv["summary"]["total_nodes"] == 1
    assert inv["nodes"][0]["content"] == large_untruncated_content

    # 1. JSON rendering
    json_report = render_list_all_report(inv, format="json")
    assert sentinel_start in json_report
    assert sentinel_end in json_report
    loaded = json.loads(json_report)
    assert loaded["nodes"][0]["content"] == large_untruncated_content

    # 2. Markdown rendering
    md_report = render_list_all_report(inv, format="markdown")
    assert "## Summary" in md_report
    assert "## Memory Nodes" in md_report
    assert "## Edges" in md_report
    assert "## Agent Sessions" in md_report
    assert "## Extraction Watermarks" in md_report
    assert "## Model Registry" in md_report
    assert sentinel_start in md_report
    assert sentinel_end in md_report
    assert large_untruncated_content in md_report

    # 3. Rich rendering
    rich_report = render_list_all_report(inv, format="rich")
    assert "Memory Nodes" in rich_report
    assert sentinel_start in rich_report
    assert sentinel_end in rich_report
    # Ensure ellipsis didn't cut off our sentinels
    assert ">>>START_ZERO_TRUNCATION" in rich_report
    assert ">>>END_ZERO_TRUNCATION" in rich_report
