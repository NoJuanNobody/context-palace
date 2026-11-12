"""
Comprehensive test suite for DualDirectionRetriever and ContextPalaceMCPServer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

import pytest

from context_palace.models import (
    AgentSession,
    ConceptNode,
    RelationEdge,
    compute_text_embedding,
)
from context_palace.retrieval import (
    CAPSULE_END_MARKER,
    CAPSULE_START_MARKER,
    DualDirectionRetriever,
)
from context_palace.mcp_server import ContextPalaceMCPServer


class FakeFederatedContextLake:
    """Lightweight in-memory fake lake implementing the FederatedContextLake interface."""

    def __init__(self) -> None:
        self.nodes: Dict[str, ConceptNode] = {}
        self.edges: Dict[str, RelationEdge] = {}
        self.sessions: Dict[str, AgentSession] = {}
        self.history_snapshots: Dict[float, Dict[str, Any]] = {}

    def add_node(
        self,
        node_id: str,
        canonical_name: str,
        category: str,
        active_value: str,
        summary: str = "",
        confidence: float = 0.9,
        pinned: bool = False,
        source_agent_id: str = "system",
        created_at: float = 100.0,
        updated_at: float = 100.0,
    ) -> ConceptNode:
        text = f"{canonical_name} {category} {active_value} {summary}"
        node = ConceptNode(
            node_id=node_id,
            canonical_name=canonical_name,
            category=category,
            active_value=active_value,
            summary=summary,
            confidence=confidence,
            pinned=pinned,
            embedding=compute_text_embedding(text),
            source_agent_id=source_agent_id,
            created_at=created_at,
            updated_at=updated_at,
            tombstoned=False,
        )
        self.nodes[node_id] = node
        return node

    def add_edge(
        self,
        edge_id: str,
        source_id: str,
        target_id: str,
        relation_type: str = "depends_on",
        valid_from: float = 100.0,
        valid_to: Optional[float] = None,
    ) -> RelationEdge:
        edge = RelationEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=1.0,
            source_agent_id="system",
            valid_from=valid_from,
            valid_to=valid_to,
        )
        self.edges[edge_id] = edge
        return edge

    def list_concepts(self, include_tombstoned: bool = False) -> List[ConceptNode]:
        if include_tombstoned:
            return list(self.nodes.values())
        return [n for n in self.nodes.values() if not n.tombstoned]

    def list_edges(self, active_only: bool = True) -> List[RelationEdge]:
        if active_only:
            return [e for e in self.edges.values() if e.valid_to is None]
        return list(self.edges.values())

    def upsert_concept(
        self,
        canonical_name: str,
        category: str = "architecture",
        value: str = "",
        summary: str = "",
        pinned: bool = False,
        source_agent_id: str = "mcp_client",
        confidence: float = 0.95,
        **kwargs: Any,
    ) -> ConceptNode:
        for n in self.nodes.values():
            if n.canonical_name == canonical_name:
                n.category = category
                n.active_value = value
                n.summary = summary
                n.pinned = pinned
                n.source_agent_id = source_agent_id
                n.confidence = confidence
                n.tombstoned = False
                n.updated_at += 10.0
                n.embedding = compute_text_embedding(f"{canonical_name} {category} {value} {summary}")
                return n
        nid = f"node_{uuid.uuid4().hex[:8]}"
        return self.add_node(
            node_id=nid,
            canonical_name=canonical_name,
            category=category,
            active_value=value,
            summary=summary,
            confidence=confidence,
            pinned=pinned,
            source_agent_id=source_agent_id,
        )

    def forget_concept(self, node_id_or_name: str, hard: bool = False) -> bool:
        target_id = None
        if node_id_or_name in self.nodes:
            target_id = node_id_or_name
        else:
            for nid, n in self.nodes.items():
                if n.canonical_name == node_id_or_name:
                    target_id = nid
                    break
        if target_id is None:
            return False
        if hard:
            del self.nodes[target_id]
        else:
            self.nodes[target_id].tombstoned = True
        return True

    def snapshot(self, as_of: float) -> Dict[str, Any]:
        if as_of in self.history_snapshots:
            return self.history_snapshots[as_of]
        nodes = [
            n.to_dict()
            for n in self.nodes.values()
            if n.created_at <= as_of and not n.tombstoned
        ]
        edges = [
            e.to_dict()
            for e in self.edges.values()
            if e.valid_from <= as_of and (e.valid_to is None or e.valid_to > as_of)
        ]
        return {"nodes": nodes, "edges": edges}

    def get_session(self, agent_id: str) -> Optional[AgentSession]:
        return self.sessions.get(agent_id)

    def list_sessions(self) -> List[AgentSession]:
        return list(self.sessions.values())


@pytest.fixture
def sample_lake() -> FakeFederatedContextLake:
    lake = FakeFederatedContextLake()
    lake.add_node(
        node_id="n_db",
        canonical_name="database_engine",
        category="architecture",
        active_value="SQLite WAL mode with bitemporal belief revisions",
        summary="Primary persistence engine",
        confidence=0.98,
        pinned=True,
        created_at=100.0,
    )
    lake.add_node(
        node_id="n_schema",
        canonical_name="belief_revision_schema",
        category="architecture",
        active_value="Tracks valid_from and valid_to timestamps per concept revision",
        summary="Bitemporal schema",
        confidence=0.92,
        pinned=False,
        created_at=110.0,
    )
    lake.add_node(
        node_id="n_auth",
        canonical_name="jwt_authentication",
        category="security",
        active_value="Ed25519 signed session tokens with 15-minute expiry",
        summary="Auth token policy",
        confidence=0.90,
        pinned=False,
        created_at=120.0,
    )
    lake.add_node(
        node_id="n_key_rot",
        canonical_name="ed25519_key_rotation",
        category="security",
        active_value="Keys rotate every 24 hours via zero-downtime JWKS endpoint",
        summary="Key rotation schedule",
        confidence=0.88,
        pinned=False,
        created_at=125.0,
    )
    lake.add_node(
        node_id="n_tui",
        canonical_name="tui_palace_keybindings",
        category="ui",
        active_value="Vim-style hjkl navigation with CRUD modals in curses/textual",
        summary="Operator TUI bindings",
        confidence=0.85,
        pinned=False,
        created_at=150.0,
    )

    lake.add_edge("e1", "n_schema", "n_db", "depends_on", valid_from=110.0)
    lake.add_edge("e2", "n_auth", "n_key_rot", "configures", valid_from=125.0)

    # Register sample agent sessions with different model windows
    lake.sessions["agent_ollama_32k"] = AgentSession(
        agent_id="agent_ollama_32k",
        harness_type="ollama",
        model_id="qwen2.5-coder:32b",
        max_context_tokens=32_768,
        used_tokens=30_500,  # Crowded: 93.08% utilization, 2,268 headroom -> needs compaction!
    )
    lake.sessions["agent_gemini_2m"] = AgentSession(
        agent_id="agent_gemini_2m",
        harness_type="antigravity",
        model_id="gemini-2.5-pro",
        max_context_tokens=2_000_000,
        used_tokens=120_000,  # Huge headroom: 1,880,000 tokens
    )
    return lake


def test_proactive_query_subgraph_1hop_and_pinned(sample_lake: FakeFederatedContextLake) -> None:
    retriever = DualDirectionRetriever(sample_lake)

    # Query specifically for jwt authentication with top_k=1 and hops=1
    # Should pick jwt_authentication or pinned database_engine, so let's test with top_k=2, hops=1
    nodes, edges = retriever.query_subgraph(
        query="jwt_authentication signed session tokens",
        top_k=2,
        hops=1,
        include_pinned=True,
    )
    node_ids = {n.node_id for n in nodes}

    # Pinned node (`n_db`) and query match (`n_auth`) are seeds;
    # 1-hop expansion pulls `n_schema` (connected to `n_db`) and `n_key_rot` (connected to `n_auth`)
    assert "n_db" in node_ids
    assert "n_auth" in node_ids
    assert "n_key_rot" in node_ids
    assert "n_schema" in node_ids

    edge_ids = {e.edge_id for e in edges}
    assert "e1" in edge_ids
    assert "e2" in edge_ids

    # Pinned concept should be ranked first
    assert nodes[0].pinned is True
    assert nodes[0].node_id == "n_db"


def test_proactive_query_subgraph_zero_hops_excludes_unmatched_neighbors(
    sample_lake: FakeFederatedContextLake,
) -> None:
    retriever = DualDirectionRetriever(sample_lake)
    nodes, edges = retriever.query_subgraph(
        query="jwt_authentication",
        top_k=1,
        hops=0,
        include_pinned=False,
    )
    assert len(nodes) == 1
    assert nodes[0].node_id == "n_auth"
    assert edges == []


def test_reactive_capsule_adapts_to_model_window_and_achieves_90pct_reduction(
    sample_lake: FakeFederatedContextLake,
) -> None:
    # Populate extra concepts so a large budget packs more than a tight budget
    for i in range(15):
        sample_lake.add_node(
            node_id=f"n_extra_{i}",
            canonical_name=f"microservice_module_{i}",
            category="architecture",
            active_value=(
                f"Service #{i} handles distributed event partition {i} with idempotent retry queues "
                f"and circuit breaker thresholds configured at 99.{i}% SLA."
            ),
            summary=f"Partition worker {i}",
            confidence=0.80,
            pinned=False,
        )

    retriever = DualDirectionRetriever(sample_lake)

    # 1. Crowded 32K local model with small explicit/adaptive budget
    crowded_session = AgentSession(
        agent_id="crowded_local",
        harness_type="ollama",
        model_id="qwen2.5-coder:7b",
        max_context_tokens=32_768,
        used_tokens=31_200,  # remaining_headroom = 1,568 -> adaptive budget = max(256, int(1568*0.12)=188) -> 256
    )
    small_capsule = retriever.build_context_capsule(
        session=crowded_session,
        task_query="microservice architecture database",
        raw_history_tokens=28_000,
    )

    # 2. 2M Gemini model with massive headroom
    large_session = sample_lake.sessions["agent_gemini_2m"]
    large_capsule = retriever.build_context_capsule(
        session=large_session,
        task_query="microservice architecture database",
        raw_history_tokens=45_000,
    )

    assert small_capsule.allocated_budget_tokens == 256
    assert small_capsule.capsule_tokens <= small_capsule.allocated_budget_tokens
    assert large_capsule.allocated_budget_tokens == 16_384
    assert len(large_capsule.nodes) > len(small_capsule.nodes)

    # Both must achieve >= 90% token reduction vs raw history tokens
    assert small_capsule.token_reduction_pct >= 90.0
    assert large_capsule.token_reduction_pct >= 90.0

    # Verify required telemetry markers in markdown payload
    assert CAPSULE_START_MARKER in small_capsule.markdown_payload
    assert CAPSULE_END_MARKER in small_capsule.markdown_payload
    assert "### Federated Context Capsule (`crowded_local` | Model: `qwen2.5-coder:7b`)" in small_capsule.markdown_payload
    assert "**Model Window:** `31,200 / 32,768 tokens" in small_capsule.markdown_payload


def test_inject_into_sidecar_is_idempotent(
    sample_lake: FakeFederatedContextLake, tmp_path: Path
) -> None:
    retriever = DualDirectionRetriever(sample_lake)
    session = sample_lake.sessions["agent_gemini_2m"]
    sidecar = tmp_path / "AGENTS.md"
    sidecar.write_text("# Project Agent Instructions\n\nAlways run pytest before committing.\n", encoding="utf-8")

    capsule_v1 = retriever.build_context_capsule(session=session, task_query="database")
    content_v1 = retriever.inject_into_sidecar(capsule_v1, str(sidecar))

    assert "# Project Agent Instructions" in content_v1
    assert content_v1.count(CAPSULE_START_MARKER) == 1
    assert content_v1.count(CAPSULE_END_MARKER) == 1
    assert "database_engine" in content_v1

    # Mutate lake and re-inject into the same sidecar file
    sample_lake.upsert_concept(
        canonical_name="new_hot_feature",
        category="task",
        value="Implement zero-copy shared memory ringbuffer",
        pinned=True,
    )
    capsule_v2 = retriever.build_context_capsule(session=session, task_query="ringbuffer")
    content_v2 = retriever.inject_into_sidecar(capsule_v2, str(sidecar))

    assert "# Project Agent Instructions" in content_v2
    assert content_v2.count(CAPSULE_START_MARKER) == 1
    assert content_v2.count(CAPSULE_END_MARKER) == 1
    assert "new_hot_feature" in content_v2


def test_mcp_initialize_and_tools_list(sample_lake: FakeFederatedContextLake) -> None:
    server = ContextPalaceMCPServer(sample_lake)

    init_resp = server.handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init_resp["result"]["serverInfo"]["name"] == "context-palace-mcp"

    list_resp = server.handle_jsonrpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = list_resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert tool_names == [
        "context_query",
        "context_inject_capsule",
        "context_upsert",
        "context_forget",
        "context_time_travel",
        "context_window_status",
    ]


def test_mcp_tool_context_query(sample_lake: FakeFederatedContextLake) -> None:
    server = ContextPalaceMCPServer(sample_lake)
    resp = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "context_query",
                "arguments": {"query": "Ed25519 session tokens", "top_k": 2, "hops": 1},
            },
        }
    )
    assert resp["result"]["isError"] is False
    data = resp["result"]["structuredContent"]
    assert data["node_count"] >= 2
    names = {n["canonical_name"] for n in data["nodes"]}
    assert "jwt_authentication" in names
    assert "ed25519_key_rotation" in names


def test_mcp_tool_context_inject_capsule(
    sample_lake: FakeFederatedContextLake, tmp_path: Path
) -> None:
    server = ContextPalaceMCPServer(sample_lake)
    sidecar_file = tmp_path / ".context_capsule.md"

    resp = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "context_inject_capsule",
                "arguments": {
                    "agent_id": "agent_ollama_32k",
                    "task_query": "SQLite bitemporal schema",
                    "sidecar_path": str(sidecar_file),
                },
            },
        }
    )
    data = resp["result"]["structuredContent"]
    assert data["target_agent_id"] == "agent_ollama_32k"
    assert data["sidecar_injected"] is True
    assert sidecar_file.exists()
    assert CAPSULE_START_MARKER in sidecar_file.read_text(encoding="utf-8")


def test_mcp_tool_context_upsert_and_forget(sample_lake: FakeFederatedContextLake) -> None:
    server = ContextPalaceMCPServer(sample_lake)

    # 1. Upsert new concept
    upsert_resp = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/call",
            "params": {
                "name": "context_upsert",
                "arguments": {
                    "canonical_name": "redis_rate_limiter",
                    "category": "config",
                    "value": "Token bucket 100 req/s per IP",
                    "summary": "Edge rate limit",
                    "pinned": True,
                    "source_agent_id": "worker_gamma",
                },
            },
        }
    )
    upserted = upsert_resp["result"]["structuredContent"]
    assert upserted["status"] == "upserted"
    assert upserted["node"]["canonical_name"] == "redis_rate_limiter"
    assert upserted["node"]["pinned"] is True

    # 2. Soft-forget concept
    forget_resp = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 13,
            "method": "tools/call",
            "params": {
                "name": "context_forget",
                "arguments": {
                    "node_id_or_name": "redis_rate_limiter",
                    "hard": False,
                },
            },
        }
    )
    forgotten = forget_resp["result"]["structuredContent"]
    assert forgotten["status"] == "tombstoned"
    assert forgotten["success"] is True

    # Confirm it no longer appears in active concepts
    active_names = {n.canonical_name for n in sample_lake.list_concepts(include_tombstoned=False)}
    assert "redis_rate_limiter" not in active_names


def test_mcp_tool_context_time_travel(sample_lake: FakeFederatedContextLake) -> None:
    server = ContextPalaceMCPServer(sample_lake)

    # At timestamp 115.0, only `n_db` (100.0) and `n_schema` (110.0) existed (`n_auth` was created at 120.0)
    resp = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 14,
            "method": "tools/call",
            "params": {
                "name": "context_time_travel",
                "arguments": {"as_of": 115.0},
            },
        }
    )
    data = resp["result"]["structuredContent"]
    assert data["as_of"] == 115.0
    assert data["node_count"] == 2
    node_ids = {n["node_id"] for n in data["nodes"]}
    assert node_ids == {"n_db", "n_schema"}
    assert data["edge_count"] == 1


def test_mcp_tool_context_window_status(sample_lake: FakeFederatedContextLake) -> None:
    server = ContextPalaceMCPServer(sample_lake)

    resp = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 15,
            "method": "tools/call",
            "params": {
                "name": "context_window_status",
                "arguments": {},
            },
        }
    )
    data = resp["result"]["structuredContent"]
    assert data["session_count"] == 2
    assert len(data["compaction_alerts"]) == 1
    assert data["compaction_alerts"][0]["agent_id"] == "agent_ollama_32k"
    assert data["compaction_alerts"][0]["utilization_pct"] > 80.0
