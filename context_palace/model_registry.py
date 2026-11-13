"""
Dynamic Model Context Window Registry and Local Agent Session Detector.
Automatically resolves LLM context window capacities (from 32K local MLX/Ollama
models up to 1M/2M Gemini/Claude/GPT-4.1 models) and inspects live agent transcripts.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from context_palace.models import AgentSession, ModelWindowSpec, estimate_tokens


BUILTIN_MODEL_SPECS: Dict[str, ModelWindowSpec] = {
    # Google Gemini family
    "gemini-2.5-pro": ModelWindowSpec(
        model_id="gemini-2.5-pro",
        canonical_name="Gemini 2.5 Pro",
        provider="google",
        max_context_tokens=2_000_000,
        max_output_tokens=65_536,
    ),
    "gemini-1.5-pro": ModelWindowSpec(
        model_id="gemini-1.5-pro",
        canonical_name="Gemini 1.5 Pro",
        provider="google",
        max_context_tokens=2_000_000,
        max_output_tokens=8_192,
    ),
    "gemini-2.5-flash": ModelWindowSpec(
        model_id="gemini-2.5-flash",
        canonical_name="Gemini 2.5 Flash",
        provider="google",
        max_context_tokens=1_000_000,
        max_output_tokens=65_536,
    ),
    "gemini-2.0-flash": ModelWindowSpec(
        model_id="gemini-2.0-flash",
        canonical_name="Gemini 2.0 Flash",
        provider="google",
        max_context_tokens=1_000_000,
        max_output_tokens=8_192,
    ),
    "gemini-4-argon": ModelWindowSpec(
        model_id="gemini-4-argon",
        canonical_name="Gemini 4 Argon",
        provider="google",
        max_context_tokens=1_000_000,
        max_output_tokens=65_536,
    ),
    "gemini-3.1-pro": ModelWindowSpec(
        model_id="gemini-3.1-pro",
        canonical_name="Gemini 3.1 Pro",
        provider="google",
        max_context_tokens=2_000_000,
        max_output_tokens=65_536,
    ),
    # Anthropic Claude family
    "claude-sonnet-4-5-1m": ModelWindowSpec(
        model_id="claude-sonnet-4-5-1m",
        canonical_name="Claude Sonnet 4.5 (1M)",
        provider="anthropic",
        max_context_tokens=1_000_000,
        max_output_tokens=64_000,
    ),
    "claude-opus-4-6-1m": ModelWindowSpec(
        model_id="claude-opus-4-6-1m",
        canonical_name="Claude Opus 4.6 (1M)",
        provider="anthropic",
        max_context_tokens=1_000_000,
        max_output_tokens=64_000,
    ),
    "claude-opus-4-6": ModelWindowSpec(
        model_id="claude-opus-4-6",
        canonical_name="Claude Opus 4.6",
        provider="anthropic",
        max_context_tokens=200_000,
        max_output_tokens=64_000,
    ),
    "claude-sonnet-4-5": ModelWindowSpec(
        model_id="claude-sonnet-4-5",
        canonical_name="Claude Sonnet 4.5",
        provider="anthropic",
        max_context_tokens=200_000,
        max_output_tokens=64_000,
    ),
    "claude-opus-4": ModelWindowSpec(
        model_id="claude-opus-4",
        canonical_name="Claude Opus 4",
        provider="anthropic",
        max_context_tokens=200_000,
        max_output_tokens=32_000,
    ),
    "claude-3-7-sonnet": ModelWindowSpec(
        model_id="claude-3-7-sonnet",
        canonical_name="Claude 3.7 Sonnet",
        provider="anthropic",
        max_context_tokens=200_000,
        max_output_tokens=64_000,
    ),
    "claude-3-5-sonnet": ModelWindowSpec(
        model_id="claude-3-5-sonnet",
        canonical_name="Claude 3.5 Sonnet",
        provider="anthropic",
        max_context_tokens=200_000,
        max_output_tokens=8_192,
    ),
    # OpenAI family
    "gpt-4.1": ModelWindowSpec(
        model_id="gpt-4.1",
        canonical_name="GPT-4.1",
        provider="openai",
        max_context_tokens=1_000_000,
        max_output_tokens=32_768,
    ),
    "gpt-4.1-mini": ModelWindowSpec(
        model_id="gpt-4.1-mini",
        canonical_name="GPT-4.1 Mini",
        provider="openai",
        max_context_tokens=1_000_000,
        max_output_tokens=32_768,
    ),
    "gpt-4o": ModelWindowSpec(
        model_id="gpt-4o",
        canonical_name="GPT-4o",
        provider="openai",
        max_context_tokens=128_000,
        max_output_tokens=16_384,
    ),
    "o3": ModelWindowSpec(
        model_id="o3",
        canonical_name="OpenAI o3",
        provider="openai",
        max_context_tokens=200_000,
        max_output_tokens=100_000,
    ),
    "o3-mini": ModelWindowSpec(
        model_id="o3-mini",
        canonical_name="OpenAI o3-mini",
        provider="openai",
        max_context_tokens=200_000,
        max_output_tokens=100_000,
    ),
    "o4-mini": ModelWindowSpec(
        model_id="o4-mini",
        canonical_name="OpenAI o4-mini",
        provider="openai",
        max_context_tokens=200_000,
        max_output_tokens=100_000,
    ),
    # xAI Grok family
    "grok-3": ModelWindowSpec(
        model_id="grok-3",
        canonical_name="Grok 3",
        provider="xai",
        max_context_tokens=131_072,
        max_output_tokens=16_384,
    ),
    "grok-4": ModelWindowSpec(
        model_id="grok-4",
        canonical_name="Grok 4",
        provider="xai",
        max_context_tokens=256_000,
        max_output_tokens=32_768,
    ),
    # DeepSeek family
    "deepseek-r1": ModelWindowSpec(
        model_id="deepseek-r1",
        canonical_name="DeepSeek R1",
        provider="deepseek",
        max_context_tokens=128_000,
        max_output_tokens=8_192,
    ),
    "deepseek-v3": ModelWindowSpec(
        model_id="deepseek-v3",
        canonical_name="DeepSeek V3",
        provider="deepseek",
        max_context_tokens=128_000,
        max_output_tokens=8_192,
    ),
    # Local Ollama / MLX models
    "qwen2.5-coder:32b": ModelWindowSpec(
        model_id="qwen2.5-coder:32b",
        canonical_name="Qwen 2.5 Coder 32B",
        provider="ollama",
        max_context_tokens=32_768,
        max_output_tokens=4_096,
    ),
    "llama3.3:70b": ModelWindowSpec(
        model_id="llama3.3:70b",
        canonical_name="Llama 3.3 70B",
        provider="ollama",
        max_context_tokens=131_072,
        max_output_tokens=8_192,
    ),
    "mlx-community/qwen2.5-7b": ModelWindowSpec(
        model_id="mlx-community/qwen2.5-7b",
        canonical_name="MLX Qwen 2.5 7B",
        provider="mlx",
        max_context_tokens=32_768,
        max_output_tokens=4_096,
    ),
}


class ModelContextRegistry:
    """
    Resolves LLM context window specs dynamically from model strings, live endpoints,
    and local agent transcript files (Antigravity, Claude Code, Codex, Ollama).
    """

    def __init__(self, custom_specs: Optional[Dict[str, ModelWindowSpec]] = None) -> None:
        self._specs: Dict[str, ModelWindowSpec] = dict(BUILTIN_MODEL_SPECS)
        if custom_specs:
            self._specs.update(custom_specs)

    def register_model(self, spec: ModelWindowSpec) -> None:
        """Register or override a model window specification."""
        self._specs[spec.model_id.lower()] = spec

    def list_known_models(self) -> List[ModelWindowSpec]:
        """Return all registered model window specifications."""
        return list(self._specs.values())

    @staticmethod
    def normalize_model_id(raw_model_id: str) -> str:
        """
        Normalize human-readable labels, provider prefixes, and date suffixes into canonical keys.
        Examples:
          - "Gemini 4 Argon (Medium)" -> "gemini-4-argon"
          - "claude-sonnet-4-5-20250929" -> "claude-sonnet-4-5"
          - "anthropic/claude-3.5-sonnet" -> "claude-3-5-sonnet"
          - "claude-opus-4-6[1m]" -> "claude-opus-4-6-1m"
        """
        if not raw_model_id:
            return "unknown"

        s = raw_model_id.strip()
        # Check for [1m] or (1m) or -1m suffix before stripping parentheses
        has_1m = bool(re.search(r"(\[1m\]|\(1m\)|[-_]1m\b)", s, flags=re.IGNORECASE))

        # Remove parenthetical effort/tier qualifiers like "(Medium)", "(High)", "(Thinking)"
        s = re.sub(r"\s*\([^)]*\)", "", s)
        s = re.sub(r"\s*\[[^\]]*\]", "", s)
        s = s.strip().lower()

        # Strip common provider prefixes unless it's mlx-community
        for prefix in ("anthropic/", "openai/", "google/", "xai/", "deepseek/", "meta-llama/", "models/"):
            if s.startswith(prefix):
                s = s[len(prefix) :]

        # Replace spaces with hyphens
        s = re.sub(r"\s+", "-", s)

        # Normalize claude version dots to hyphens (e.g., claude-3.5-sonnet -> claude-3-5-sonnet, claude-sonnet-4.5 -> claude-sonnet-4-5)
        if "claude" in s:
            s = re.sub(r"(\d)\.(\d)", r"\1-\2", s)

        # Strip trailing 8-digit date stamps like -20250929 or -20241022
        s = re.sub(r"-20\d{6}$", "", s)
        # Strip trailing :latest for ollama if not in builtin
        if s.endswith(":latest"):
            s = s[:-7]

        if has_1m and not s.endswith("-1m"):
            s = f"{s}-1m"

        return s

    def resolve_model_window(self, model_id: str, fetch_live: bool = False) -> ModelWindowSpec:
        """
        Resolve the context window capacity for a given model identifier.
        Optionally queries OpenRouter or local Ollama if `fetch_live=True`.
        """
        raw_lower = (model_id or "").strip().lower()
        if raw_lower in self._specs:
            return self._specs[raw_lower]

        norm = self.normalize_model_id(model_id)
        if norm in self._specs:
            return self._specs[norm]

        # Check prefix/substring matches against known catalog keys
        for key, spec in self._specs.items():
            if norm.startswith(key) or key in norm:
                # Respect explicit 1m suffix if present on a Claude model
                if norm.endswith("-1m") and spec.max_context_tokens < 1_000_000:
                    return ModelWindowSpec(
                        model_id=norm,
                        canonical_name=f"{spec.canonical_name} (1M)",
                        provider=spec.provider,
                        max_context_tokens=1_000_000,
                        max_output_tokens=spec.max_output_tokens,
                        source="builtin_catalog",
                    )
                return spec

        if fetch_live:
            live_spec = self._try_fetch_live(model_id, norm)
            if live_spec is not None:
                self._specs[norm] = live_spec
                return live_spec

        # Heuristic fallback based on model family / tags
        if "1m" in norm or "1000k" in norm:
            provider = "anthropic" if "claude" in norm else "google" if "gemini" in norm else "custom"
            return ModelWindowSpec(
                model_id=norm,
                canonical_name=model_id.strip() or norm,
                provider=provider,
                max_context_tokens=1_000_000,
                max_output_tokens=32_768,
                source="heuristic",
            )
        if "2m" in norm or "gemini-1.5-pro" in norm or "gemini-2.5-pro" in norm:
            return ModelWindowSpec(
                model_id=norm,
                canonical_name=model_id.strip() or norm,
                provider="google",
                max_context_tokens=2_000_000,
                max_output_tokens=32_768,
                source="heuristic",
            )
        if "gemini" in norm:
            return ModelWindowSpec(
                model_id=norm,
                canonical_name=model_id.strip() or norm,
                provider="google",
                max_context_tokens=1_000_000,
                max_output_tokens=32_768,
                source="heuristic",
            )
        if "claude" in norm:
            return ModelWindowSpec(
                model_id=norm,
                canonical_name=model_id.strip() or norm,
                provider="anthropic",
                max_context_tokens=200_000,
                max_output_tokens=16_384,
                source="heuristic",
            )
        if any(tag in norm for tag in ("mlx", "qwen", "ollama", "phi-", "mistral", "codellama")):
            return ModelWindowSpec(
                model_id=norm,
                canonical_name=model_id.strip() or norm,
                provider="mlx" if "mlx" in norm else "ollama",
                max_context_tokens=32_768,
                max_output_tokens=4_096,
                source="heuristic",
            )

        return ModelWindowSpec(
            model_id=norm,
            canonical_name=model_id.strip() or norm,
            provider="unknown",
            max_context_tokens=128_000,
            max_output_tokens=8_192,
            source="default_fallback",
        )

    def _try_fetch_live(self, raw_model_id: str, norm_id: str) -> Optional[ModelWindowSpec]:
        """Attempt fast 1.5s lookup against local Ollama or OpenRouter public catalog."""
        # 1. Try local Ollama tags if it looks like a local tag or ollama
        try:
            req = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                for m in data.get("models", []):
                    name = (m.get("name") or "").lower()
                    if name == norm_id or name.startswith(norm_id):
                        return ModelWindowSpec(
                            model_id=norm_id,
                            canonical_name=m.get("name", norm_id),
                            provider="ollama",
                            max_context_tokens=32_768,
                            max_output_tokens=4_096,
                            source="live_ollama",
                        )
        except Exception:
            pass

        # 2. Try OpenRouter models endpoint
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/models", method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                for m in data.get("data", []):
                    mid = (m.get("id") or "").lower()
                    if mid == raw_lower_or_norm(raw_model_id, norm_id) or mid.endswith(f"/{norm_id}"):
                        ctx = int(m.get("context_length") or 128_000)
                        provider = mid.split("/")[0] if "/" in mid else "openrouter"
                        return ModelWindowSpec(
                            model_id=norm_id,
                            canonical_name=m.get("name") or norm_id,
                            provider=provider,
                            max_context_tokens=ctx,
                            max_output_tokens=int(m.get("top_provider", {}).get("max_completion_tokens") or 8_192),
                            source="live_openrouter",
                        )
        except Exception:
            pass

        return None

    def detect_agent_session_from_file(self, path: str, agent_id: Optional[str] = None) -> AgentSession:
        """
        Inspect a `.jsonl`, `.json`, or `.md` transcript file to detect the agent harness,
        active model, token usage, turn count, and remaining context headroom.
        """
        file_path = Path(path).expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Transcript file not found: {path}")

        stat = file_path.stat()
        byte_size = stat.st_size
        mtime = stat.st_mtime
        path_str = str(file_path)
        path_lower = path_str.lower()

        # Initial harness guess from path
        harness_type = "custom"
        if "antigravity" in path_lower or "gemini" in path_lower:
            harness_type = "antigravity"
        elif "claude" in path_lower:
            harness_type = "claude_code"
        elif "codex" in path_lower:
            harness_type = "codex"
        elif "ollama" in path_lower:
            harness_type = "ollama"

        detected_model: Optional[str] = None
        explicit_max_tokens: Optional[int] = None
        used_tokens = 0
        turn_count = 0
        session_meta: Dict[str, Any] = {}

        raw_text = file_path.read_text(encoding="utf-8", errors="replace")

        if file_path.suffix.lower() == ".jsonl" or (
            raw_text.lstrip().startswith("{") and "\n" in raw_text.strip()
        ):
            lines = [ln for ln in raw_text.splitlines() if ln.strip()]
            turn_count = len(lines)
            for line in lines:
                used_tokens += estimate_tokens(line)
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue

                # Harness detection from JSONL schema
                if "step_index" in obj or obj.get("type") in (
                    "PLANNER_RESPONSE",
                    "USER_INPUT",
                    "USER_SETTINGS_CHANGE",
                ):
                    harness_type = "antigravity"
                elif "sessionId" in obj or "cwd" in obj or obj.get("type") in ("user", "assistant"):
                    if harness_type == "custom":
                        harness_type = "claude_code"

                # Direct model field extraction
                for key in ("model", "model_id", "modelName", "selectedModel"):
                    val = obj.get(key)
                    if isinstance(val, str) and val.strip():
                        detected_model = val.strip()

                # Nested message.model (Claude Code assistant turns often store model inside obj["message"]["model"])
                msg = obj.get("message")
                if isinstance(msg, dict):
                    m_val = msg.get("model") or msg.get("model_id")
                    if isinstance(m_val, str) and m_val.strip():
                        detected_model = m_val.strip()

                # Antigravity USER_SETTINGS_CHANGE parsing: "Model Selection ... to Gemini 4 Argon (Medium)"
                if obj.get("type") == "USER_SETTINGS_CHANGE" or "Model Selection" in str(obj.get("content", "")):
                    content_str = str(obj.get("content", ""))
                    m = re.search(r"Model Selection.*?to\s+([^\n\r]+)", content_str, flags=re.IGNORECASE)
                    if m:
                        detected_model = m.group(1).strip()

                # Optional explicit max_context_tokens in metadata
                if isinstance(obj.get("max_context_tokens"), int):
                    explicit_max_tokens = obj["max_context_tokens"]
        else:
            # Plain JSON or Markdown transcript
            used_tokens = estimate_tokens(raw_text)
            turn_count = max(1, len([ln for ln in raw_text.splitlines() if ln.strip()]))
            if file_path.suffix.lower() == ".json":
                try:
                    obj = json.loads(raw_text)
                    if isinstance(obj, dict):
                        for key in ("model", "model_id", "modelName"):
                            if isinstance(obj.get(key), str):
                                detected_model = obj[key]
                        if isinstance(obj.get("messages"), list):
                            turn_count = len(obj["messages"])
                except Exception:
                    pass

        # Fallback model per harness if none found in transcript
        if not detected_model:
            if harness_type == "antigravity":
                detected_model = "gemini-4-argon"
            elif harness_type == "claude_code":
                detected_model = "claude-sonnet-4-5"
            elif harness_type == "codex":
                detected_model = "gpt-4o"
            elif harness_type == "ollama":
                detected_model = "qwen2.5-coder:32b"
            else:
                detected_model = "gpt-4o"

        spec = self.resolve_model_window(detected_model)
        max_ctx = explicit_max_tokens if explicit_max_tokens is not None else spec.max_context_tokens

        if not agent_id:
            agent_id = self._derive_agent_id(file_path, harness_type)

        session_meta["canonical_model_name"] = spec.canonical_name
        session_meta["provider"] = spec.provider
        session_meta["raw_detected_model"] = detected_model

        try:
            from context_palace.summarizer import extract_transcript_summary

            t_info = extract_transcript_summary(raw_text)
            session_meta["topic_summary"] = t_info.get("topic_summary", "Active Coding Session")
            session_meta["initial_topic_summary"] = t_info.get("initial_topic_summary", "")
            session_meta["latest_prompt"] = t_info.get("latest_prompt", "")
            session_meta["first_prompt"] = t_info.get("first_prompt", "")
        except Exception:
            pass

        return AgentSession(
            agent_id=agent_id,
            harness_type=harness_type,
            model_id=spec.model_id,
            max_context_tokens=max_ctx,
            used_tokens=used_tokens,
            transcript_path=path_str,
            last_processed_turn=turn_count,
            last_byte_offset=byte_size,
            updated_at=mtime,
            metadata=session_meta,
        )

    @staticmethod
    def _derive_agent_id(file_path: Path, harness_type: str) -> str:
        """Derive a concise human-readable agent_id from a transcript path."""
        parts = file_path.parts
        # For Antigravity: .../brain/<conversation-id>/.system_generated/logs/transcript.jsonl
        if "brain" in parts:
            idx = parts.index("brain")
            if idx + 1 < len(parts):
                conv_id = parts[idx + 1][:8]
                prefix = "agy" if harness_type == "antigravity" else harness_type
                return f"{prefix}-{conv_id}"
        stem = file_path.stem
        if stem == "transcript" and file_path.parent.name:
            return f"{harness_type}-{file_path.parent.name[:8]}"
        return f"{harness_type}-{stem[:12]}"

    def discover_local_agents(
        self,
        search_dirs: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[AgentSession]:
        """
        Scan local agent directories (e.g. ~/.gemini/antigravity-cli/brain and ~/.claude/projects)
        for recent primary transcript files (excluding internal chunk files and headless subagents)
        and return their live AgentSession states.
        """
        if search_dirs is None:
            default_dirs = [
                os.path.expanduser("~/.gemini/antigravity-cli/brain"),
                os.path.expanduser("~/.claude/projects"),
            ]
            search_dirs = [d for d in default_dirs if os.path.isdir(d)]

        candidates: List[Path] = []
        for d in search_dirs:
            root = Path(d).expanduser()
            if not root.exists():
                continue
            if root.is_file():
                candidates.append(root)
                continue
            for p in root.rglob("*.jsonl"):
                # Skip internal chunk slices and transcript_full duplicates
                if "chunks" in p.parts or p.name == "transcript_full.jsonl":
                    continue
                # For Antigravity brain directories, skip headless subagent transcripts (no USER_INPUT)
                if "brain" in p.parts and p.name == "transcript.jsonl":
                    try:
                        txt = p.read_text(encoding="utf-8", errors="ignore")
                        if '"USER_INPUT"' not in txt:
                            continue
                    except OSError:
                        continue
                candidates.append(p)

        # Sort by modification time descending
        candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)

        sessions: List[AgentSession] = []
        seen_ids = set()
        for cand in candidates:
            if len(sessions) >= limit:
                break
            try:
                sess = self.detect_agent_session_from_file(str(cand))
                if sess.agent_id in seen_ids:
                    continue
                seen_ids.add(sess.agent_id)
                sessions.append(sess)
            except Exception:
                continue

        return sessions


def raw_lower_or_norm(raw_model_id: str, norm_id: str) -> str:
    return (raw_model_id or norm_id).strip().lower()
