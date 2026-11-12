"""
Comprehensive unit test suite for ConversationalMemoryPalace (Context Palace v2.0).
Tests:
- Deterministic natural language & slash command intent classification
- Natural language Q&A over populated memories with node citations and 1-hop subgraph provenance
- Conversational 'remember that ...' creating a new node in ContextLake with auto-linking
- Conversational 'update ... to ...' superseding an old node and creating a SUPERSEDES edge
- Conversational 'forget ...' invalidating a node (valid_to / valid_until set)
- Conversational time-travel ('as_of') returning historical beliefs prior to supersession
- Model context window and token budget inspection
- Multi-turn interactive REPL (run_repl) with mocked input/output
- Turn history and session token accounting in agent_sessions
"""

from __future__ import annotations

import time
from typing import Any, Dict, List
import pytest

from context_palace.consensus import ConsensusEngine
from context_palace.conversation import ConversationTurn, ConversationalMemoryPalace
from context_palace.lake import ContextLake
from context_palace.model_registry import ModelContextRegistry
from context_palace.models import ConceptNode
from context_palace.retrieval import DualDirectionRetriever


@pytest.fixture
def memory_lake() -> ContextLake:
    """Fixture providing a fresh in-memory ContextLake."""
    lake = ContextLake(":memory:")
    yield lake
    lake.close()


@pytest.fixture
def chat_palace(memory_lake: ContextLake) -> ConversationalMemoryPalace:
    """Fixture providing a ConversationalMemoryPalace connected to an in-memory lake."""
    return ConversationalMemoryPalace(
        lake=memory_lake,
        agent_name="test-chat-agent",
        model_id="gemini-2.5-pro",
        session_id="test-session-01",
    )


# =============================================================================
# 1. Intent Classification Tests
# =============================================================================


def test_classify_intent_remember_variants(chat_palace: ConversationalMemoryPalace) -> None:
    """Test intent parsing for various remember commands and natural phrases."""
    # Slash command
    intent, params = chat_palace.classify_intent("/remember user prefers dark mode")
    assert intent == "remember"
    assert params["content"] == "user prefers dark mode"
    assert params["memory_type"] == "preference"

    # Natural language with "that"
    intent, params = chat_palace.classify_intent("remember that our database is PostgreSQL 16")
    assert intent == "remember"
    assert "PostgreSQL 16" in params["content"]
    assert params["memory_type"] == "architecture"

    # Natural language save
    intent, params = chat_palace.classify_intent("save that the team decided to use Rust for core engine")
    assert intent == "remember"
    assert "Rust" in params["content"]
    assert params["memory_type"] == "decision"

    # Note command with explicit bracketed type
    intent, params = chat_palace.classify_intent("/note [constraint] maximum latency is 50ms")
    assert intent == "remember"
    assert params["memory_type"] == "constraint"
    assert "maximum latency is 50ms" in params["content"]


def test_classify_intent_forget_variants(chat_palace: ConversationalMemoryPalace) -> None:
    """Test intent parsing for forget commands and deletion phrases."""
    intent, params = chat_palace.classify_intent("/forget node-1234")
    assert intent == "forget"
    assert params["target"] == "node-1234"

    intent, params = chat_palace.classify_intent("delete node node-abc567")
    assert intent == "forget"
    assert params["target"] == "node-abc567"

    intent, params = chat_palace.classify_intent("remove memory about legacy Redis cache")
    assert intent == "forget"
    assert params["target"] == "legacy Redis cache"


def test_classify_intent_update_variants(chat_palace: ConversationalMemoryPalace) -> None:
    """Test intent parsing for update commands and phrases."""
    intent, params = chat_palace.classify_intent("/update node-abc to PostgreSQL 16")
    assert intent == "update"
    assert params["target"] == "node-abc"
    assert params["new_content"] == "PostgreSQL 16"

    intent, params = chat_palace.classify_intent("change primary database to SQLite with WAL mode")
    assert intent == "update"
    assert params["target"] == "primary database"
    assert params["new_content"] == "SQLite with WAL mode"

    intent, params = chat_palace.classify_intent("update auth strategy with Ed25519 tokens")
    assert intent == "update"
    assert params["target"] == "auth strategy"
    assert params["new_content"] == "Ed25519 tokens"


def test_classify_intent_list_and_windows_and_as_of(chat_palace: ConversationalMemoryPalace) -> None:
    """Test intent parsing for /list, /windows, and /as-of commands."""
    assert chat_palace.classify_intent("/list")[0] == "list"
    assert chat_palace.classify_intent("show all memories")[0] == "list"
    assert chat_palace.classify_intent("what is in the lake?")[0] == "list"

    assert chat_palace.classify_intent("/windows")[0] == "window_status"
    assert chat_palace.classify_intent("context window")[0] == "window_status"
    assert chat_palace.classify_intent("token budget")[0] == "window_status"

    intent, params = chat_palace.classify_intent("/as-of 2026-11-12T12:00:00Z auth strategy")
    assert intent == "time_travel"
    assert params["as_of"] == "2026-11-12T12:00:00Z"
    assert params["query"] == "auth strategy"

    intent, params = chat_palace.classify_intent("what did we believe at 1700000000.0 regarding database?")
    assert intent == "time_travel"
    assert params["as_of"] == "1700000000.0"
    assert "database" in params["query"]


def test_classify_intent_default_query(chat_palace: ConversationalMemoryPalace) -> None:
    """Test that natural language questions default to query intent."""
    intent, params = chat_palace.classify_intent("How does our consensus algorithm work?")
    assert intent == "query"
    assert params["query"] == "How does our consensus algorithm work?"


# =============================================================================
# 2. Natural Language Q&A with Citations and 1-Hop Subgraph Provenance
# =============================================================================


def test_natural_language_qa_with_subgraph_provenance(memory_lake: ContextLake) -> None:
    """Verify natural language Q&A grounds answers in populated memories and includes 1-hop relations."""
    n1 = memory_lake.create_concept(
        canonical_name="Auth Service",
        category="architecture",
        value="Ed25519 authenticated JWT service listening on port 8081",
        summary="Ed25519 Auth Service on port 8081",
        confidence=0.96,
        source_agent_id="architect",
        timestamp=100.0,
    )
    n2 = memory_lake.create_concept(
        canonical_name="User Database",
        category="database",
        value="PostgreSQL 16 replica with connection pooling",
        summary="User DB on PostgreSQL 16",
        confidence=0.92,
        source_agent_id="dba",
        timestamp=100.0,
    )
    edge = memory_lake.add_edge(
        source_id=n1.node_id,
        target_id=n2.node_id,
        relation_type="queries",
        weight=0.9,
        source_agent_id="system",
        timestamp=100.0,
    )

    palace = ConversationalMemoryPalace(lake=memory_lake, model_id="gemini-2.5-pro")

    response = palace.ask("How does the Auth Service authenticate requests?")
    assert response["intent"] == "query"
    assert n1.node_id[:8] in response["answer"]
    assert "Ed25519" in response["answer"]

    # Verify citation format: [node_id[:8]] (source_agent, tier, confidence)
    expected_citation_pattern = f"[{n1.node_id[:8]}] ({n1.source_agent_id}, {n1.category}, {n1.confidence:.2f})"
    assert expected_citation_pattern in response["answer"]

    # Verify cited nodes list
    cited_ids = [n["node_id"] for n in response["cited_nodes"]]
    assert n1.node_id in cited_ids

    # Verify 1-hop subgraph includes induced edges
    subgraph = response["subgraph"]
    edge_types = [e["relation_type"] for e in subgraph["edges"]]
    assert "queries" in edge_types

    # Verify history tracking
    assert len(palace.history) == 2
    assert palace.history[0].role == "user"
    assert palace.history[1].role == "assistant"
    assert n1.node_id in palace.history[1].cited_node_ids


# =============================================================================
# 3. Conversational "remember that ..." creating new node in ContextLake
# =============================================================================


def test_conversational_remember_creates_node_and_links(chat_palace: ConversationalMemoryPalace) -> None:
    """Verify conversational 'remember that ...' creates a new node in ContextLake and discovers relations."""
    # Pre-populate a related node
    chat_palace.lake.create_concept(
        canonical_name="Web Framework",
        category="architecture",
        value="Next.js App Router for frontend UI",
        timestamp=10.0,
    )

    response = chat_palace.ask("remember that our primary UI library is Tailwind CSS for Next.js App Router")
    assert response["intent"] == "remember"
    assert len(response["cited_nodes"]) >= 1

    # Extract created node
    created_dict = response["cited_nodes"][0]
    new_node_id = created_dict["node_id"]
    assert new_node_id[:8] in response["answer"]

    # Verify node exists in ContextLake
    stored = chat_palace.lake.get_concept(new_node_id)
    assert stored is not None
    assert "Tailwind CSS" in stored.active_value
    assert stored.category in ("architecture", "preference", "fact")

    # Verify belief revision exists in ContextLake
    revs = chat_palace.lake.get_belief_history(new_node_id)
    assert len(revs) == 1
    assert revs[0].valid_to is None
    assert revs[0].is_active is True


# =============================================================================
# 4. Conversational "update ... to ..." superseding and creating SUPERSEDES edge
# =============================================================================


def test_conversational_update_supersedes_node_and_creates_edge(chat_palace: ConversationalMemoryPalace) -> None:
    """Verify conversational 'update ... to ...' supersedes the old node and creates a SUPERSEDES edge."""
    old_node = chat_palace.lake.create_concept(
        canonical_name="Database Engine",
        category="config",
        value="MySQL 8.0 on AWS RDS",
        confidence=0.85,
        source_agent_id="dev-1",
        timestamp=100.0,
    )

    response = chat_palace.ask("update Database Engine to PostgreSQL 16 with pgvector on Supabase")
    assert response["intent"] == "update"
    assert "--[SUPERSEDES]-->" in response["answer"]

    # Check cited nodes has [new_node, old_node]
    assert len(response["cited_nodes"]) == 2
    new_node_dict = response["cited_nodes"][0]
    new_node_id = new_node_dict["node_id"]
    assert new_node_id != old_node.node_id

    # Verify SUPERSEDES edge exists in ContextLake
    edges = chat_palace.lake.list_edges(active_only=True)
    supersedes_edges = [
        e for e in edges
        if e.source_id == new_node_id and e.target_id == old_node.node_id and e.relation_type == "SUPERSEDES"
    ]
    assert len(supersedes_edges) == 1
    assert supersedes_edges[0].weight == 1.0

    # Old node should be soft-deleted / tombstoned from active concepts
    assert chat_palace.lake.get_concept(old_node.node_id, include_tombstoned=False) is None
    old_tombstoned = chat_palace.lake.get_concept(old_node.node_id, include_tombstoned=True)
    assert old_tombstoned is not None
    assert old_tombstoned.tombstoned is True

    # New node is active
    new_active = chat_palace.lake.get_concept(new_node_id)
    assert new_active is not None
    assert "PostgreSQL 16 with pgvector" in new_active.active_value


# =============================================================================
# 5. Conversational "forget ..." invalidating a node (valid_to set)
# =============================================================================


def test_conversational_forget_invalidates_node(chat_palace: ConversationalMemoryPalace) -> None:
    """Verify conversational 'forget ...' soft-deletes the node and closes its active belief revision."""
    legacy_node = chat_palace.lake.create_concept(
        canonical_name="Deprecated Caching Layer",
        category="architecture",
        value="Memcached instance on 11211",
        timestamp=50.0,
    )

    response = chat_palace.ask("forget Deprecated Caching Layer")
    assert response["intent"] == "forget"
    assert legacy_node.node_id[:8] in response["answer"]
    assert "valid_until" in response["answer"] or "invalid" in response["answer"]

    # Concept should no longer be in active concepts
    assert chat_palace.lake.get_concept(legacy_node.node_id, include_tombstoned=False) is None

    # Tombstoned version still exists in lake bitemporal store
    tombstoned = chat_palace.lake.get_concept(legacy_node.node_id, include_tombstoned=True)
    assert tombstoned is not None
    assert tombstoned.tombstoned is True

    # Active belief revision must be closed (valid_to is set)
    history = chat_palace.lake.get_belief_history(legacy_node.node_id)
    assert len(history) >= 1
    assert history[0].valid_to is not None
    assert history[0].is_active is False


# =============================================================================
# 6. Conversational Time-Travel ("as_of") Prior to Supersession
# =============================================================================


def test_conversational_time_travel_returns_historical_beliefs(memory_lake: ContextLake) -> None:
    """Verify time-travel queries reconstruct what was believed at as_of vs current state."""
    # At t=100.0, create initial belief
    c1 = memory_lake.create_concept(
        canonical_name="Cache Architecture",
        category="architecture",
        value="Redis 6 in-memory single node",
        confidence=0.85,
        source_agent_id="dev-1",
        timestamp=100.0,
    )

    # At t=200.0, update to Dragonfly cluster
    memory_lake.update_concept(
        node_id_or_name=c1.node_id,
        new_value="Dragonfly Cluster v1.12 with multi-threaded shards",
        confidence=0.98,
        source_agent_id="operator",
        timestamp=200.0,
    )

    palace = ConversationalMemoryPalace(lake=memory_lake)

    # Query as of t=150.0 (between 100.0 and 200.0)
    resp = palace.ask("/as-of 150.0 Cache Architecture")
    assert resp["intent"] == "time_travel"
    assert "Redis 6 in-memory single node" in resp["answer"]
    assert "Dragonfly Cluster" in resp["answer"]  # evolution section reports update
    assert c1.node_id[:8] in resp["answer"]

    # Query without explicit slash command using as_of parameter
    resp_param = palace.ask("what is cache architecture?", as_of="150.0")
    assert resp_param["intent"] == "time_travel"
    assert "Redis 6 in-memory single node" in resp_param["answer"]


# =============================================================================
# 7. Window Status and Token Budget Accounting
# =============================================================================


def test_window_status_and_token_accounting(chat_palace: ConversationalMemoryPalace) -> None:
    """Verify /windows reports accurate model context capacity, used tokens, and capsule headroom."""
    # Ask a few turns to accumulate tokens
    chat_palace.ask("remember that we use Python 3.15 with pytest 9")
    chat_palace.ask("what test framework do we use?")

    resp = chat_palace.ask("/windows")
    assert resp["intent"] == "window_status"
    assert "gemini-2.5-pro" in resp["answer"]
    assert "Capacity" in resp["answer"]
    assert "Used Tokens" in resp["answer"]
    assert "Remaining Headroom" in resp["answer"]
    assert "Adaptive Capsule Budget" in resp["answer"]

    # Verify session persisted in ContextLake
    session = chat_palace.lake.get_agent_session(chat_palace.session_id)
    assert session is not None
    assert session.used_tokens > 0
    assert session.max_context_tokens == 2_000_000
    assert session.last_processed_turn >= 6


# =============================================================================
# 8. Interactive Multi-Turn REPL (run_repl)
# =============================================================================


def test_run_repl_interactive_commands(memory_lake: ContextLake) -> None:
    """Verify run_repl processes /help, /remember, /list, /windows, /history, and exits on /exit."""
    palace = ConversationalMemoryPalace(lake=memory_lake)

    simulated_inputs = [
        "/help",
        "/remember [config] default port is 8080",
        "what is the default port?",
        "/list",
        "/windows",
        "/history",
        "/exit",
    ]
    input_iterator = iter(simulated_inputs)

    outputs: List[str] = []

    def mock_input(prompt: str = "") -> str:
        return next(input_iterator)

    def mock_output(msg: str) -> None:
        outputs.append(msg)

    exit_code = palace.run_repl(input_fn=mock_input, output_fn=mock_output)
    assert exit_code == 0

    combined_output = "\n".join(outputs)
    assert "Available Commands" in combined_output
    assert "COMMITTED" in combined_output or "committed" in combined_output
    assert "8080" in combined_output
    assert "Federated Context Lake Overview" in combined_output or "Active Concepts" in combined_output
    assert "Model Context Window" in combined_output
    assert "Conversation History" in combined_output
    assert "Exiting conversational REPL. Goodbye!" in combined_output


def test_run_repl_handles_eof_gracefully(memory_lake: ContextLake) -> None:
    """Verify run_repl terminates cleanly with exit code 0 when EOFError is raised."""
    palace = ConversationalMemoryPalace(lake=memory_lake)

    def mock_input_eof(prompt: str = "") -> str:
        raise EOFError()

    outputs: List[str] = []
    exit_code = palace.run_repl(input_fn=mock_input_eof, output_fn=outputs.append)
    assert exit_code == 0
    assert any("Exiting" in out for out in outputs)


# =============================================================================
# 9. Format Turn Rich and LLM Callback Support
# =============================================================================


def test_format_turn_rich_display(chat_palace: ConversationalMemoryPalace) -> None:
    """Verify format_turn_rich renders citations, provenance, and token badges."""
    resp = chat_palace.ask("remember that our CI/CD provider is GitHub Actions")
    formatted = chat_palace.format_turn_rich(resp)
    assert "[REMEMBER]" in formatted
    assert "GitHub Actions" in formatted
    assert "Session Tokens:" in formatted


def test_custom_llm_callback_integration(memory_lake: ContextLake) -> None:
    """Verify custom LLM callback is invoked when provided for natural language query synthesis."""
    memory_lake.create_concept(
        canonical_name="Secret Key",
        category="security",
        value="Stored in AWS Secrets Manager",
        timestamp=10.0,
    )

    def custom_llm(query: str, cited: List[Dict[str, Any]]) -> str:
        return f"LLM-SYNTHESIS: For query '{query}', found {len(cited)} secrets."

    palace = ConversationalMemoryPalace(lake=memory_lake, llm_callback=custom_llm)
    resp = palace.ask("where is the secret key stored?")
    assert resp["intent"] == "query"
    assert resp["answer"].startswith("LLM-SYNTHESIS:")
    assert "where is the secret key stored?" in resp["answer"]
