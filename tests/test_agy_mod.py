"""
Tests for the Context Palace Antigravity Mod (`agy_hook.py` and `install-agy-mod` CLI).
"""

import json
from pathlib import Path
import pytest

from context_palace.agy_hook import handle_agy_hook_payload
from context_palace.cli import main
from context_palace.lake import ContextLake
from context_palace.models import AgentSession


def test_agy_hook_session_start_injects_capsule(tmp_path: Path) -> None:
    db_path = str(tmp_path / "hook_test.db")
    lake = ContextLake(db_path)
    lake.create_concept(
        canonical_name="federated_memory_topology",
        category="architecture",
        value="1M working memory + SQLite WAL global graph lake",
        pinned=True,
    )

    payload = {
        "conversationId": "3ad38af4-75a2-47be-8ddd-2bdb785bc70d",
        "workspacePaths": [str(tmp_path)],
        "modelName": "Gemini 4 Argon (Medium)",
        "lastUserInput": "Check federated memory architecture",
        "hookEventName": "SessionStart",
    }

    resp = handle_agy_hook_payload(payload, db_path=db_path)
    assert "injectSteps" in resp
    assert len(resp["injectSteps"]) == 1
    sys_msg = resp["injectSteps"][0]["systemMessage"]
    assert "federated_memory_topology" in sys_msg["systemMessage"]
    assert "Context Palace Attached" in sys_msg["metadata"]["user_markdown"]
    assert "gemini-4-argon" in sys_msg["metadata"]["user_markdown"]


def test_agy_hook_pre_invocation_compaction_warning(tmp_path: Path) -> None:
    db_path = str(tmp_path / "hook_compact.db")
    lake = ContextLake(db_path)
    lake.create_concept(
        canonical_name="core_constraint",
        category="constraint",
        value="Preserve pinned subgraphs on compaction",
        pinned=True,
    )
    # Seed an agent session at 85% utilization
    lake.upsert_agent_session(
        AgentSession(
            agent_id="agy-99887766",
            harness_type="antigravity",
            model_id="qwen2.5-coder:32b",
            max_context_tokens=32_768,
            used_tokens=28_500,
        )
    )

    payload = {
        "conversationId": "99887766-0000-0000-0000-000000000000",
        "modelName": "qwen2.5-coder:32b",
        "lastUserInput": "Continue coding",
        "hookEventName": "PreInvocation",
    }
    resp = handle_agy_hook_payload(payload, db_path=db_path)
    assert len(resp["injectSteps"]) == 1
    assert "Compaction Alert" in resp["injectSteps"][0]["systemMessage"]["metadata"]["user_markdown"]


def test_agy_hook_post_invocation_extracts_transcript_deltas(tmp_path: Path) -> None:
    db_path = str(tmp_path / "hook_post.db")
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        '{"step_index": 1, "type": "USER_INPUT", "content": "Storage Engine: SQLite WAL Bitemporal"}\n'
        '{"step_index": 2, "type": "PLANNER_RESPONSE", "content": "Gateway routes to ContextLake"}\n',
        encoding="utf-8",
    )

    payload = {
        "conversationId": "11223344-5566-7788-99aa-bbccddeeff00",
        "transcriptPath": str(transcript),
        "modelName": "gemini-4-argon",
        "hookEventName": "PostInvocation",
    }
    resp = handle_agy_hook_payload(payload, db_path=db_path)
    assert resp == {"injectSteps": [], "terminationBehavior": ""}

    lake = ContextLake(db_path)
    concepts = lake.list_concepts()
    assert any("sqlite" in c.active_value.lower() for c in concepts)


def test_install_agy_mod_cli_creates_plugin_mcp_hooks_and_skills(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg_root = tmp_path / "gemini_config"
    rc = main(["install-agy-mod", "--config-root", str(cfg_root)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"

    plugin_dir = cfg_root / "plugins" / "context-palace"
    assert (plugin_dir / "plugin.json").exists()
    assert (plugin_dir / "mcp_config.json").exists()
    assert (plugin_dir / "hooks.json").exists()
    assert (plugin_dir / "rules" / "AGENTS.md").exists()
    assert (plugin_dir / "skills" / "context-palace" / "SKILL.md").exists()
    assert (cfg_root / "skills" / "context-palace" / "SKILL.md").exists()

    mcp_cfg = json.loads((cfg_root / "mcp_config.json").read_text(encoding="utf-8"))
    assert "context_palace" in mcp_cfg["mcpServers"]
