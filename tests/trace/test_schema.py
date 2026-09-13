from dataclasses import replace

import pytest

from moescope.trace import (
    TRACE_SCHEMA_VERSION,
    RoutingRule,
    RoutingTrace,
    TokenRoute,
    TraceValidationError,
)


@pytest.fixture
def minimal_trace() -> RoutingTrace:
    return RoutingTrace(
        schema_version=TRACE_SCHEMA_VERSION,
        trace_kind="synthetic",
        model_name="moescope-test",
        model_revision="test-revision",
        phase="prefill",
        layer_index=0,
        step_index=0,
        num_routed_experts=4,
        num_shared_experts=2,
        routed_experts_per_token=2,
        routing_rule=RoutingRule(
            score_function="precomputed",
            selection_method="top_k",
            weight_normalization="not_normalized",
        ),
        tokens=(
            TokenRoute(
                token_index=0,
                expert_ids=(0, 2),
                weights=(0.6, 0.2),
            ),
        ),
    )


def test_trace_round_trips_through_deterministic_json(
    minimal_trace: RoutingTrace,
) -> None:
    serialized = minimal_trace.to_json()

    assert RoutingTrace.from_json(serialized) == minimal_trace
    assert RoutingTrace.from_json(serialized).to_json() == serialized


def test_trace_counts_routed_and_shared_assignments_separately(
    minimal_trace: RoutingTrace,
) -> None:
    assert minimal_trace.routed_assignment_count == 2
    assert minimal_trace.shared_assignment_count == 2


def test_trace_rejects_unsupported_schema_version(
    minimal_trace: RoutingTrace,
) -> None:
    with pytest.raises(TraceValidationError, match="unsupported trace schema"):
        replace(minimal_trace, schema_version="2.0.0")


def test_trace_rejects_wrong_number_of_selections(
    minimal_trace: RoutingTrace,
) -> None:
    route = TokenRoute(
        token_index=0,
        expert_ids=(0,),
        weights=(1.0,),
    )

    with pytest.raises(TraceValidationError, match="select exactly 2"):
        replace(minimal_trace, tokens=(route,))


def test_trace_rejects_out_of_range_routed_expert(
    minimal_trace: RoutingTrace,
) -> None:
    route = TokenRoute(
        token_index=0,
        expert_ids=(0, 4),
        weights=(0.6, 0.2),
    )

    with pytest.raises(TraceValidationError, match=r"outside \[0, 4\)"):
        replace(minimal_trace, tokens=(route,))


def test_token_route_rejects_duplicate_routed_experts() -> None:
    with pytest.raises(TraceValidationError, match="duplicate routed expert"):
        TokenRoute(
            token_index=0,
            expert_ids=(1, 1),
            weights=(0.6, 0.4),
        )


def test_token_route_rejects_non_finite_weight() -> None:
    with pytest.raises(TraceValidationError, match="must be finite"):
        TokenRoute(
            token_index=0,
            expert_ids=(0, 1),
            weights=(float("nan"), 0.4),
        )


def test_parser_rejects_unknown_fields(minimal_trace: RoutingTrace) -> None:
    payload = minimal_trace.to_dict()
    payload["benchmark_latency_ms"] = 1.2

    with pytest.raises(TraceValidationError, match="unknown fields"):
        RoutingTrace.from_dict(payload)
