"""
Federated Context Lake & TUI CRUD Memory Palace (context-palace).
Shared data models and interface contracts for the 4-worker Argon Swarm.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import math


@dataclass
class ModelWindowSpec:
    """Represents a resolved LLM context window specification."""
    model_id: str
    canonical_name: str
    provider: str
    max_context_tokens: int
    max_output_tokens: int = 8192
    source: str = "builtin_catalog"  # "live_openrouter", "live_ollama", "session_metadata", "builtin_catalog"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentSession:
    """Represents a connected local coding agent session and its live context window usage."""
    agent_id: str
    harness_type: str  # "antigravity", "claude_code", "codex", "ollama", "custom"
    model_id: str
    max_context_tokens: int
    used_tokens: int = 0
    transcript_path: Optional[str] = None
    last_processed_turn: int = 0
    last_byte_offset: int = 0
    updated_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_headroom(self) -> int:
        return max(0, self.max_context_tokens - self.used_tokens)

    @property
    def utilization_pct(self) -> float:
        if self.max_context_tokens <= 0:
            return 0.0
        return round((self.used_tokens / self.max_context_tokens) * 100.0, 2)

    @property
    def needs_compaction(self) -> bool:
        return self.utilization_pct >= 80.0

    def adaptive_capsule_budget(self, ratio: float = 0.12, min_tokens: int = 256, max_cap: int = 16384) -> int:
        """Compute how many tokens a reactive Context Capsule should use for this model's headroom."""
        raw = int(self.remaining_headroom * ratio)
        if self.remaining_headroom < min_tokens:
            return max(64, self.remaining_headroom)
        return max(min_tokens, min(raw, max_cap))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["remaining_headroom"] = self.remaining_headroom
        d["utilization_pct"] = self.utilization_pct
        d["needs_compaction"] = self.needs_compaction
        return d


@dataclass
class ConceptNode:
    """Represents a concept, entity, or architectural fact in the Federated Context Lake."""
    node_id: str
    canonical_name: str
    category: str  # e.g., "architecture", "config", "constraint", "entity", "decision", "task"
    active_value: str
    summary: str = ""
    confidence: float = 0.9
    pinned: bool = False
    embedding: List[float] = field(default_factory=list)
    source_agent_id: str = "system"
    created_at: float = 0.0
    updated_at: float = 0.0
    tombstoned: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BeliefRevision:
    """Immutable bitemporal belief record tracking how an agent's or system's beliefs evolve."""
    revision_id: str
    node_id: str
    value: str
    confidence: float
    source_agent_id: str
    provenance_type: str = "agent_turn"  # "user_explicit", "verified_test", "operator_crud", "agent_turn", "model_inference"
    turn_id: int = 0
    valid_from: float = 0.0
    valid_to: Optional[float] = None  # None means currently active belief
    superseded_by: Optional[str] = None

    @property
    def is_active(self) -> bool:
        return self.valid_to is None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RelationEdge:
    """Directed typed edge between two ConceptNodes in the knowledge graph."""
    edge_id: str
    source_id: str
    target_id: str
    relation_type: str  # e.g., "depends_on", "configures", "implements", "conflicts_with", "owned_by"
    weight: float = 1.0
    source_agent_id: str = "system"
    valid_from: float = 0.0
    valid_to: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConflictResolutionEvent:
    """Audit record emitted when ConsensusEngine resolves conflicting beliefs across agents."""
    event_id: str
    node_id: str
    winning_revision_id: str
    losing_revision_id: str
    winning_agent_id: str
    losing_agent_id: str
    winning_score: float
    losing_score: float
    reason: str
    resolved_at: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContextCapsule:
    """Compact subgraph payload produced by Dual-Direction Retrieval for injection into an agent's working context."""
    target_agent_id: str
    target_model_id: str
    max_context_tokens: int
    remaining_headroom: int
    allocated_budget_tokens: int
    capsule_tokens: int
    raw_history_tokens_equivalent: int
    token_reduction_pct: float
    nodes: List[ConceptNode]
    edges: List[RelationEdge]
    markdown_payload: str
    as_of_timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_agent_id": self.target_agent_id,
            "target_model_id": self.target_model_id,
            "max_context_tokens": self.max_context_tokens,
            "remaining_headroom": self.remaining_headroom,
            "allocated_budget_tokens": self.allocated_budget_tokens,
            "capsule_tokens": self.capsule_tokens,
            "raw_history_tokens_equivalent": self.raw_history_tokens_equivalent,
            "token_reduction_pct": self.token_reduction_pct,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "markdown_payload": self.markdown_payload,
            "as_of_timestamp": self.as_of_timestamp,
        }


def estimate_tokens(text: str) -> int:
    """Fast deterministic token estimator (~4 chars/token or word-piece approximation)."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4.0))


def compute_text_embedding(text: str, dim: int = 64) -> List[float]:
    """
    Deterministic character/word n-gram hashing embedding for zero-dependency local vector similarity.
    Produces an L2-normalized vector of length `dim`.
    """
    vec = [0.0] * dim
    if not text:
        return vec
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
    tokens = [t for t in cleaned.split() if t]
    features: List[str] = list(tokens)
    for token in tokens:
        padded = f"<{token}>"
        for i in range(len(padded) - 2):
            features.append(padded[i : i + 3])
    if not features:
        return vec
    for feat in features:
        h = 2166136261
        for ch in feat:
            h = (h ^ ord(ch)) * 16777619 & 0xFFFFFFFF
        idx = h % dim
        sign = 1.0 if ((h >> 16) & 1) == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [round(v / norm, 6) for v in vec]
    return vec


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))
