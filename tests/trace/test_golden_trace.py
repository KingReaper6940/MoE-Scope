from pathlib import Path

from moescope.trace import RoutingTrace, summarize_routing

FIXTURE_PATH = (
    Path(__file__).parents[2]
    / "examples"
    / "traces"
    / "four-token-top2.json"
)


def load_golden_trace() -> RoutingTrace:
    return RoutingTrace.from_json(FIXTURE_PATH.read_text())


def test_four_token_trace_matches_the_hand_worked_histogram() -> None:
    summary = summarize_routing(load_golden_trace())

    assert summary.num_tokens == 4
    assert summary.routed_experts_per_token == 2
    assert summary.routed_histogram == (2, 2, 3, 1)
    assert summary.active_routed_experts == (0, 1, 2, 3)
    assert summary.inactive_routed_experts == ()


def test_four_token_trace_keeps_routed_and_shared_work_separate() -> None:
    summary = summarize_routing(load_golden_trace())

    assert summary.routed_assignment_count == 8
    assert summary.shared_assignment_count == 8
    assert summary.total_assignment_count == 16


def test_four_token_trace_serialization_is_canonical() -> None:
    trace = load_golden_trace()

    assert trace.to_json() + "\n" == FIXTURE_PATH.read_text()


def test_four_token_summary_is_deterministic() -> None:
    trace = load_golden_trace()
    first_summary = summarize_routing(trace).to_json()
    reparsed_trace = RoutingTrace.from_json(trace.to_json())
    second_summary = summarize_routing(reparsed_trace).to_json()

    assert second_summary == first_summary
