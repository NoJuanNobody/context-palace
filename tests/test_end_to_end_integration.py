"""
End-to-end integration test verifying all 4 Argon Swarm subsystems working together:
ModelContextRegistry + ContextLake + IncrementalDeltaExtractor + ConsensusEngine +
DualDirectionRetriever + ContextPalaceMCPServer + MemoryPalaceTUI.
"""

from pathlib import Path
from context_palace import (
    ModelContextRegistry,
    ContextLake,
    ConsensusEngine,
    IncrementalDeltaExtractor,
    DualDirectionRetriever,
    ContextPalaceMCPServer,
    MemoryPalaceTUI,
)


def test_full_federated_context_lake_lifecycle(tmp_path: Path) -> None:
    db_file = tmp_path / "federated_lake.db"
    lake = ContextLake(str(db_file))
    registry = ModelContextRegistry()
    consensus = ConsensusEngine()
    extractor = IncrementalDeltaExtractor(consensus_engine=consensus)
    retriever = DualDirectionRetriever(lake=lake)
    mcp = ContextPalaceMCPServer(lake=lake, retriever=retriever, registry=registry)
    tui = MemoryPalaceTUI(lake=lake, retriever=retriever, registry=registry)

    # 1. Create synthetic transcripts for two decoupled agents on different models
    agy_transcript = tmp_path / "antigravity_transcript.jsonl"
    agy_transcript.write_text(
        '{"step_index": 1, "type": "USER_INPUT", "content": "The user changed setting Model Selection from None to Gemini 4 Argon (Medium). Database: SQLite WAL. Port: 8080"}\n'
        '{"step_index": 2, "type": "PLANNER_RESPONSE", "content": "Architecture uses Federated Context Lake. Cache depends on Vector Deduplicator."}\n',
        encoding="utf-8",
    )

    claude_transcript = tmp_path / "claude_session.jsonl"
    claude_transcript.write_text(
        '{"sessionId": "claude-1", "model": "claude-sonnet-4-5", "message": {"role": "assistant", "content": "Port: 9090. Service implements DualDirectionRetriever."}}\n',
        encoding="utf-8",
    )

    # 2. Auto-detect model context windows from transcripts
    agy_session = registry.detect_agent_session_from_file(str(agy_transcript), agent_id="agy-agent-1")
    claude_session = registry.detect_agent_session_from_file(str(claude_transcript), agent_id="claude-agent-1")

    assert agy_session.max_context_tokens == 1_000_000
    assert claude_session.max_context_tokens == 200_000

    # Reset last_processed_turn to 0 before first delta ingestion
    agy_session.last_processed_turn = 0
    claude_session.last_processed_turn = 0
    lake.upsert_agent_session(agy_session)
    lake.upsert_agent_session(claude_session)

    # 3. Ingest turns from Agent 1 (user_explicit + agent_turn) at t=1000
    res1 = extractor.ingest_turns(
        lake=lake,
        session=agy_session,
        turns=[
            {"turn_id": 1, "role": "user", "content": "Database: SQLite WAL\nPort: 8080"},
            {"turn_id": 2, "role": "assistant", "content": "Architecture: Federated Context Lake"},
        ],
        timestamp=1000.0,
    )
    assert res1["processed_turns"] == 2
    assert res1["created_nodes"] >= 2

    # 4. Ingest conflicting turn from Agent 2 (lower provenance agent_turn tries to change Port to 9090 at t=1010)
    res2 = extractor.ingest_turns(
        lake=lake,
        session=claude_session,
        turns=[
            {"turn_id": 1, "role": "assistant", "content": "Port: 9090\nRetrieval: DualDirectionRetriever"},
        ],
        timestamp=1010.0,
    )
    assert res2["processed_turns"] == 1
    assert res2["conflicts_resolved"] >= 1

    # Verify user_explicit "8080" won consensus over agent_turn "9090"
    port_node = lake.get_concept("port")
    assert port_node is not None
    assert "8080" in port_node.active_value

    # 5. Operator overrides Port via TUI CRUD 'u' at t=2000
    tui.execute_key_action(
        "u",
        {
            "node_id_or_name": "port",
            "new_value": "8443 (TLS)",
            "timestamp": 2000.0,
            "pinned": True,
        },
    )
    assert lake.get_concept("port").active_value == "8443 (TLS)"

    # 6. Time-travel query to t=1500 verifies historical belief "8080"
    snap_1500 = lake.snapshot(as_of=1500.0)
    port_at_1500 = [
        n for n in snap_1500["nodes"]
        if (n.canonical_name if hasattr(n, "canonical_name") else n["canonical_name"]) == "port"
    ][0]
    val_1500 = port_at_1500.active_value if hasattr(port_at_1500, "active_value") else port_at_1500["active_value"]
    assert "8080" in val_1500

    # 7. Build adaptive Context Capsule and verify >= 90% token reduction
    capsule = retriever.build_context_capsule(
        session=claude_session,
        task_query="database port architecture",
        raw_history_tokens=25_000,
    )
    assert capsule.token_reduction_pct >= 90.0
    assert "8443 (TLS)" in capsule.markdown_payload

    # 8. Verify MCP JSON-RPC window status & TUI dashboard rendering
    rpc_resp = mcp.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 42,
            "method": "tools/call",
            "params": {"name": "context_window_status", "arguments": {}},
        }
    )
    assert "result" in rpc_resp
    dashboard_text = tui.render_dashboard()
    assert "Context Palace" in dashboard_text
    assert "agy-agent-1" in dashboard_text
    assert "claude-agent-1" in dashboard_text
