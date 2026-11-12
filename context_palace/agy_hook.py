#!/usr/bin/env python3
"""
Antigravity Lifecycle Hook Handler for Context Palace (`context-palace`).
Implements the official Antigravity `hooks.json` stdin/stdout JSON contract for:
- `SessionStart`: Auto-detects the active model context window, registers the session in the
  Federated Context Lake, and injects a compact Context Capsule (`injectSteps`).
- `PreInvocation`: Checks live context window utilization and injects pinned/relevant subgraphs
  or compaction warnings when needed.
- `PostInvocation`: Incrementally extracts newly appended turns from `transcriptPath` into the
  SQLite WAL Context Lake (`~/.context_palace/context_lake.db`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

from context_palace.consensus import ConsensusEngine
from context_palace.extractor import IncrementalDeltaExtractor
from context_palace.lake import ContextLake
from context_palace.model_registry import ModelContextRegistry
from context_palace.models import AgentSession
from context_palace.retrieval import DualDirectionRetriever


def get_default_db_path() -> str:
    return os.environ.get(
        "CONTEXT_PALACE_DB",
        str(Path("~/.context_palace/context_lake.db").expanduser()),
    )


def handle_agy_hook_payload(
    payload: Dict[str, Any],
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process an Antigravity hook payload (camelCase JSON on stdin) and return the
    contract-compliant JSON response for stdout.
    """
    resolved_db = db_path or get_default_db_path()
    event_name = str(payload.get("hookEventName") or "")
    conv_id = str(payload.get("conversationId") or "unknown")
    transcript_path = payload.get("transcriptPath")
    model_name = str(payload.get("modelName") or "auto")
    last_user_input = str(payload.get("lastUserInput") or "")

    lake = ContextLake(resolved_db)
    registry = ModelContextRegistry()
    retriever = DualDirectionRetriever(lake=lake)
    extractor = IncrementalDeltaExtractor(consensus_engine=ConsensusEngine())

    agent_id = f"agy-{conv_id[:8]}" if conv_id != "unknown" else "agy-session"

    # Resolve or detect session and its model context window
    existing_session = lake.get_agent_session(agent_id)
    if transcript_path and os.path.exists(str(transcript_path)):
        detected = registry.detect_agent_session_from_file(
            str(transcript_path), agent_id=agent_id
        )
        if model_name and model_name.lower() not in ("auto", "none", ""):
            spec = registry.resolve_model_window(model_name)
            detected.model_id = spec.model_id
            detected.max_context_tokens = spec.max_context_tokens
        detected.last_processed_turn = (
            existing_session.last_processed_turn if existing_session is not None else 0
        )
        session = lake.upsert_agent_session(detected)
    else:
        effective_model = (
            model_name
            if model_name.lower() not in ("auto", "none", "")
            else (existing_session.model_id if existing_session else "gemini-4-argon")
        )
        spec = registry.resolve_model_window(effective_model)
        session = AgentSession(
            agent_id=agent_id,
            harness_type="antigravity",
            model_id=spec.model_id,
            max_context_tokens=spec.max_context_tokens,
            used_tokens=existing_session.used_tokens if existing_session else 0,
            transcript_path=str(transcript_path) if transcript_path else None,
            last_processed_turn=existing_session.last_processed_turn if existing_session else 0,
            updated_at=time.time(),
        )
        lake.upsert_agent_session(session)

    if event_name == "SessionStart":
        capsule = retriever.build_context_capsule(
            session=session,
            task_query=last_user_input or "architecture constraint pinned",
        )
        if not capsule.nodes:
            return {}
        user_md = (
            f"**Context Palace Attached** (`{session.model_id}` | "
            f"`{session.used_tokens:,} / {session.max_context_tokens:,}` tokens | "
            f"`{len(capsule.nodes)}` active facts injected)"
        )
        return {
            "injectSteps": [
                {
                    "systemMessage": {
                        "systemMessage": capsule.markdown_payload,
                        "metadata": {
                            "user_markdown": user_md,
                        },
                    }
                }
            ]
        }

    if event_name == "PreInvocation":
        # Only inject a warning or refresh capsule if the session is approaching compaction (>= 80%)
        if session.needs_compaction:
            capsule = retriever.build_context_capsule(
                session=session,
                task_query=last_user_input,
            )
            warn_msg = (
                f"[Context Palace Alert] Session `{session.agent_id}` on `{session.model_id}` "
                f"is at {session.utilization_pct:.1f}% context capacity "
                f"({session.used_tokens:,}/{session.max_context_tokens:,} tokens).\n"
                f"{capsule.markdown_payload}"
            )
            return {
                "injectSteps": [
                    {
                        "systemMessage": {
                            "systemMessage": warn_msg,
                            "metadata": {
                                "user_markdown": (
                                    f"⚠️ **Context Palace Compaction Alert**: `{session.utilization_pct:.1f}%` "
                                    f"of `{session.model_id}` window used (`{session.remaining_headroom:,}` headroom left)."
                                )
                            },
                        }
                    }
                ],
                "terminationBehavior": "",
            }
        return {"injectSteps": [], "terminationBehavior": ""}

    if event_name == "PostInvocation":
        # Incrementally extract new turns from transcriptPath into the Context Lake
        if transcript_path and os.path.exists(str(transcript_path)):
            turns_list: List[Dict[str, Any]] = []
            try:
                with open(str(transcript_path), "r", encoding="utf-8", errors="replace") as fh:
                    for idx, raw_line in enumerate(fh, start=1):
                        raw_line = raw_line.strip()
                        if not raw_line:
                            continue
                        try:
                            obj = json.loads(raw_line)
                            if isinstance(obj, dict):
                                step_type = str(obj.get("type") or "")
                                if step_type and step_type not in ("USER_INPUT", "PLANNER_RESPONSE", ""):
                                    continue
                                content_val = obj.get("content") or ""
                                if content_val and len(str(content_val)) <= 6000:
                                    turns_list.append(
                                        {
                                            "turn_id": idx,
                                            "role": obj.get("source") or step_type or "agent_turn",
                                            "content": str(content_val),
                                        }
                                    )
                        except Exception:
                            continue
            except OSError:
                turns_list = []

            if turns_list:
                extractor.ingest_turns(lake=lake, session=session, turns=turns_list)
                lake.upsert_agent_session(session)
        return {"injectSteps": [], "terminationBehavior": ""}

    if event_name == "PostToolUse":
        return {}

    return {}


def main() -> int:
    raw_stdin = sys.stdin.read().strip()
    if not raw_stdin:
        sys.stdout.write("{}\n")
        return 0
    try:
        payload = json.loads(raw_stdin)
        if not isinstance(payload, dict):
            sys.stdout.write("{}\n")
            return 0
        resp = handle_agy_hook_payload(payload)
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()
        return 0
    except Exception:
        # Never block or crash the Antigravity agent loop if a hook encounters an error
        sys.stdout.write("{}\n")
        sys.stdout.flush()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
