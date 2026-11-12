"""
Seed a throwaway Context Lake with the synthetic sample data from the v2 UI brief, so the
TUI can be previewed / screenshotted without touching ~/.context_palace.

    python scripts/seed_demo_lake.py /tmp/cp_demo.db
    context-palace --db /tmp/cp_demo.db tui
"""

from __future__ import annotations

import random
import sys
import time
from typing import Optional

from context_palace.lake import ContextLake
from context_palace.models import AgentSession, ConflictResolutionEvent


def seed(db_path: str, now: Optional[float] = None) -> ContextLake:
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

    # Belief revisions spread over the afternoon so the LAKE sparkline has a shape.
    lake.update_concept("auth_strategy", "HS256 JWT, 1h TTL", confidence=0.70, provenance_type="agent_turn",
                        source_agent_id="antigravity-3ad38af4", timestamp=start + 3600)
    lake.update_concept("auth_strategy", "Ed25519 JWT, 15m TTL", confidence=0.86, provenance_type="verified_test",
                        source_agent_id="claude-code-dev", timestamp=start + 2 * 3600)
    lake.update_concept("auth_strategy", "Use Ed25519 + WebAuthn passkeys", confidence=0.97,
                        provenance_type="operator_crud", source_agent_id="operator_cli", timestamp=start + 4.5 * 3600)
    for k in range(60):
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


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/context_palace_demo.db"
    seed(path)
    print(path)
