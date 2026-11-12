"""
ConsensusEngine for resolving conflicting beliefs across multi-agent turns in context-palace.
"""

from __future__ import annotations

import math
import time
import uuid
from typing import Dict, Optional, Tuple

from context_palace.models import BeliefRevision, ConflictResolutionEvent

PROVENANCE_WEIGHTS: Dict[str, float] = {
    "operator_crud": 1.00,
    "user_explicit": 0.95,
    "verified_test": 0.90,
    "agent_turn": 0.70,
    "model_inference": 0.50,
}


class ConsensusEngine:
    """
    Deterministic multi-agent belief conflict resolution engine.
    Scores revisions using provenance authority, explicit confidence,
    temporal recency, and multi-agent vote support.
    """

    def __init__(self, provenance_weights: Optional[Dict[str, float]] = None) -> None:
        self.provenance_weights: Dict[str, float] = dict(PROVENANCE_WEIGHTS)
        if provenance_weights:
            self.provenance_weights.update(provenance_weights)

    def score_revision(
        self,
        revision: BeliefRevision,
        current_time: Optional[float] = None,
        vote_count: int = 1,
    ) -> float:
        """
        Compute a deterministic composite consensus score combining:
          1. Provenance weight (PROVENANCE_WEIGHTS.get(revision.provenance_type, 0.6) * 0.45)
          2. Explicit confidence (revision.confidence * 0.30)
          3. Recency factor (0.15 scaled so newer valid_from timestamps score higher than older ones)
          4. Multi-agent vote bonus (min(0.25, (max(1, vote_count) - 1) * 0.08))
        """
        prov_raw = self.provenance_weights.get(revision.provenance_type, 0.60)
        provenance_component = prov_raw * 0.45

        confidence_clamped = max(0.0, min(1.0, float(revision.confidence)))
        confidence_component = confidence_clamped * 0.30

        valid_from = max(0.0, float(revision.valid_from))
        if current_time is not None:
            ref_time = max(valid_from, float(current_time))
            age_seconds = max(0.0, ref_time - valid_from)
            recency_ratio = 1.0 / (1.0 + (age_seconds / 3600.0))
        else:
            if valid_from <= 0.0:
                recency_ratio = 0.5
            else:
                # Smoothly monotonic across both small synthetic timestamps and Unix epochs
                recency_ratio = 0.5 + 0.5 * (math.atan(math.log1p(valid_from) / 5.0) / (math.pi / 2.0))
        recency_component = 0.15 * recency_ratio

        effective_votes = max(1, int(vote_count))
        vote_bonus = min(0.25, (effective_votes - 1) * 0.08)

        total_score = provenance_component + confidence_component + recency_component + vote_bonus
        return round(total_score, 6)

    @staticmethod
    def _lookup_votes(revision: BeliefRevision, vote_counts: Optional[Dict[str, int]]) -> int:
        if not vote_counts:
            return 1
        for key in (revision.revision_id, revision.value, revision.source_agent_id):
            if key in vote_counts:
                return max(1, int(vote_counts[key]))
        return 1

    def resolve(
        self,
        existing_revision: BeliefRevision,
        candidate_revision: BeliefRevision,
        vote_counts: Optional[Dict[str, int]] = None,
        resolved_at: Optional[float] = None,
    ) -> Tuple[BeliefRevision, BeliefRevision, ConflictResolutionEvent]:
        """
        Compare existing_revision and candidate_revision using score_revision.
        Gives candidate_revision a slight recency tie-breaker if timestamps are equal
        and candidate provenance weight is >= existing provenance weight.
        Returns (winner, loser, ConflictResolutionEvent).
        """
        now = resolved_at if resolved_at is not None else time.time()
        ref_time = max(
            float(existing_revision.valid_from),
            float(candidate_revision.valid_from),
            float(now) if resolved_at is not None else 0.0,
        )

        existing_votes = self._lookup_votes(existing_revision, vote_counts)
        candidate_votes = self._lookup_votes(candidate_revision, vote_counts)

        existing_score = self.score_revision(
            existing_revision,
            current_time=ref_time,
            vote_count=existing_votes,
        )
        candidate_score = self.score_revision(
            candidate_revision,
            current_time=ref_time,
            vote_count=candidate_votes,
        )

        existing_prov = self.provenance_weights.get(existing_revision.provenance_type, 0.60)
        candidate_prov = self.provenance_weights.get(candidate_revision.provenance_type, 0.60)

        tie_broken = False
        if (
            abs(candidate_score - existing_score) < 1e-6
            and candidate_revision.valid_from >= existing_revision.valid_from
            and candidate_prov >= existing_prov
        ):
            candidate_score = round(candidate_score + 0.0005, 6)
            tie_broken = True

        if candidate_score >= existing_score:
            winner = candidate_revision
            loser = existing_revision
            winning_score = candidate_score
            losing_score = existing_score
        else:
            winner = existing_revision
            loser = candidate_revision
            winning_score = existing_score
            losing_score = candidate_score

        winner_votes = candidate_votes if winner is candidate_revision else existing_votes
        loser_votes = existing_votes if winner is candidate_revision else candidate_votes
        winner_prov = self.provenance_weights.get(winner.provenance_type, 0.60)
        loser_prov = self.provenance_weights.get(loser.provenance_type, 0.60)

        reasons = []
        if winner_prov > loser_prov:
            reasons.append(
                f"higher provenance authority ({winner.provenance_type}={winner_prov:.2f} > {loser.provenance_type}={loser_prov:.2f})"
            )
        if winner_votes > loser_votes:
            reasons.append(f"multi-agent consensus votes ({winner_votes} vs {loser_votes})")
        if winner.confidence > loser.confidence:
            reasons.append(f"higher confidence ({winner.confidence:.2f} > {loser.confidence:.2f})")
        if winner.valid_from > loser.valid_from:
            reasons.append("more recent timestamp")
        if tie_broken and not reasons:
            reasons.append("candidate recency tie-breaker on equal timestamp and provenance")
        if not reasons:
            reasons.append(f"composite score ({winning_score:.4f} vs {losing_score:.4f})")

        reason_str = "; ".join(reasons)
        node_id = winner.node_id or loser.node_id

        event = ConflictResolutionEvent(
            event_id=f"cre_{uuid.uuid4().hex[:12]}",
            node_id=node_id,
            winning_revision_id=winner.revision_id,
            losing_revision_id=loser.revision_id,
            winning_agent_id=winner.source_agent_id,
            losing_agent_id=loser.source_agent_id,
            winning_score=winning_score,
            losing_score=losing_score,
            reason=reason_str,
            resolved_at=float(now),
        )
        return winner, loser, event
