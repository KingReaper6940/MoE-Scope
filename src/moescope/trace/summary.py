"""Deterministic summaries derived from validated routing traces."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .schema import RoutingTrace


@dataclass(frozen=True, slots=True)
class RoutingSummary:
    """Small set of facts needed to inspect a routing event."""

    num_tokens: int
    num_routed_experts: int
    num_shared_experts: int
    routed_experts_per_token: int
    routed_assignment_count: int
    shared_assignment_count: int
    routed_histogram: tuple[int, ...]
    active_routed_experts: tuple[int, ...]
    inactive_routed_experts: tuple[int, ...]

    @property
    def total_assignment_count(self) -> int:
        return self.routed_assignment_count + self.shared_assignment_count

    def to_dict(self) -> dict[str, object]:
        return {
            "num_tokens": self.num_tokens,
            "num_routed_experts": self.num_routed_experts,
            "num_shared_experts": self.num_shared_experts,
            "routed_experts_per_token": self.routed_experts_per_token,
            "routed_assignment_count": self.routed_assignment_count,
            "shared_assignment_count": self.shared_assignment_count,
            "total_assignment_count": self.total_assignment_count,
            "routed_histogram": list(self.routed_histogram),
            "active_routed_experts": list(self.active_routed_experts),
            "inactive_routed_experts": list(self.inactive_routed_experts),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def summarize_routing(trace: RoutingTrace) -> RoutingSummary:
    """Count routed assignments without folding in shared-expert execution."""

    histogram = [0] * trace.num_routed_experts
    for token in trace.tokens:
        for expert_id in token.expert_ids:
            histogram[expert_id] += 1

    routed_assignment_count = sum(histogram)
    if routed_assignment_count != trace.routed_assignment_count:
        raise RuntimeError(
            "routing summary disagrees with the validated trace assignment count"
        )

    active = tuple(
        expert_id
        for expert_id, count in enumerate(histogram)
        if count > 0
    )
    inactive = tuple(
        expert_id
        for expert_id, count in enumerate(histogram)
        if count == 0
    )

    return RoutingSummary(
        num_tokens=trace.num_tokens,
        num_routed_experts=trace.num_routed_experts,
        num_shared_experts=trace.num_shared_experts,
        routed_experts_per_token=trace.routed_experts_per_token,
        routed_assignment_count=routed_assignment_count,
        shared_assignment_count=trace.shared_assignment_count,
        routed_histogram=tuple(histogram),
        active_routed_experts=active,
        inactive_routed_experts=inactive,
    )
