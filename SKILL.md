---
name: context-palace
description: Federated Context Lake & TUI CRUD Memory Palace with Automatic Model Context Window Detection, Incremental Transcript Delta Extraction, Bitemporal Belief Revision, and Dual-Direction Context Capsule Injection. Use when inspecting, editing, compacting, or sharing working memory across local coding agents (Antigravity, Claude Code, Codex, Ollama).
---

# Context Palace (`context-palace`)

**Context Palace** is a Federated Context Lake and Rich Terminal User Interface (TUI) CRUD Memory Palace that solves the *Working Memory Bottleneck* across heterogeneous local coding agents. Instead of forcing agents to re-read bloated 100k+ token raw conversation logs or relying on opaque black-box RAG vector stores, `context-palace` continuously extracts typed concept nodes, relationships, and bitemporal belief revisions into a shared SQLite WAL knowledge graph while giving human operators full **Create, Read, Update, Delete, Pin, and Time-Travel** authority.

---

## Core Capabilities

1. **Auto-Detected Model Context Windows (`ModelContextRegistry`)**:
   - Automatically detects each connected agent's active LLM (`gemini-2.5-pro`, `claude-sonnet-4`, `gpt-4o`, `qwen2.5-coder:7b`, etc.) from session JSONL transcripts, local Ollama `/api/show` endpoints, OpenRouter model metadata, or a built-in catalog.
   - Computes live token utilization (`used_tokens / max_context_tokens`), remaining headroom, and an **Adaptive Capsule Budget** (`12%` of free headroom, clamped to `[256, 16384]` tokens).
   - Flags sessions exceeding `80%` utilization with **`COMPACT NOW`** alerts in the TUI.

2. **Bitemporal Knowledge Graph & Consensus Engine (`ContextLake` + `ConsensusEngine`)**:
   - Stores `ConceptNode`, `BeliefRevision` (`valid_from` / `valid_to`), `RelationEdge`, and `ConflictResolutionEvent` records in SQLite WAL.
   - Resolves multi-agent belief conflicts deterministically via provenance authority (`user_explicit > verified_test > operator_crud > agent_turn > model_inference`), recency decay, and confidence weighting.

3. **Dual-Direction Retrieval & Adaptive Context Capsules (`DualDirectionRetriever`)**:
   - **Proactive Push**: Automatically includes pinned architectural invariants and high-centrality concepts.
   - **Reactive Pull**: Hybrid BM25/FTS + 64-dim cosine similarity + 1-hop Personalized PageRank expansion tailored to the agent's current query and remaining model context headroom.

4. **Rich TUI CRUD Inspector (`MemoryPalaceTUI`)**:
   - Interactive dashboard and deterministic key action dispatcher (`[c]` Create, `[r]` Read/History, `[u]` Update, `[d]` Delete/Evict, `[p]` Pin & Inject Capsule, `[m]` Refresh Model Windows, `[t]` Time-Travel, `[q]` Quit).

---

## CLI Quick Reference

```bash
# 1. Render the Memory Palace TUI dashboard once (or omit --once for interactive mode)
context-palace tui --once

# 2. Scan local agent transcripts, auto-detect model context windows, and ingest turn deltas
context-palace scan-agents --ingest

# 3. Inspect resolved context window limits for specific models
context-palace models claude-sonnet-4 gemini-2.5-pro qwen2.5-coder:7b

# 4. CRUD Operations on the Federated Memory Lake
context-palace crud create --name "auth_strategy" --category "architecture" --value "Use Ed25519 JWTs with 15m TTL" --pinned
context-palace crud list
context-palace crud read --name "auth_strategy"
context-palace crud update --name "auth_strategy" --value "Use Ed25519 + WebAuthn passkeys"
context-palace crud delete --name "auth_strategy"

# 5. Build and inject an adaptive Context Capsule fitted to an agent's model headroom
context-palace inject --agent-id "agy-worker-1" --query "authentication middleware" --output .context_palace_capsule.md

# 6. Bitemporal Time-Travel Snapshot
context-palace time-travel --as-of 1762970000.0

# 7. Launch the JSON-RPC stdio MCP Server
context-palace mcp
```
