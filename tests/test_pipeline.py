from dataclasses import replace
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import pytest

from moescope.routing import generate_trace, route_scores
from moescope.trace import RoutingTrace, TokenRoute, TraceValidationError, summarize_routing
from moescope.workloads import WorkloadConfig, artifact_hash, compile_workload
from moescope.replay import benchmark, compare_outputs, grouped_replay, make_inputs, reference_replay

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "examples/traces/four-token-top2.json"


def test_original_fixture_can_be_derived_from_scores():
    expected = RoutingTrace.from_json(GOLDEN.read_text())
    actual = route_scores([[.6, .1, .2, .1], [.05, .7, .2, .05],
                           [.4, .1, .45, .05], [.2, .3, .1, .4]],
                          top_k=2, shared_experts=2, logits=False, normalize=False,
                          name=expected.model_name, revision=expected.model_revision)
    assert actual.to_json() == expected.to_json()


def test_stable_softmax_and_tie_rule():
    trace = route_scores([[10000., 10000., -10000.]], top_k=2)
    assert trace.tokens[0].expert_ids == (0, 1)
    assert trace.tokens[0].weights == (.5, .5)


@pytest.mark.parametrize("scores,k", [([], 1), ([[1], [1, 2]], 1),
                                    ([[float('inf')]], 1), ([[True]], 1), ([[1]], 2)])
def test_invalid_scores(scores, k):
    with pytest.raises(ValueError):
        route_scores(scores, top_k=k)


@pytest.mark.parametrize("distribution", ["balanced", "random", "skewed"])
def test_generator_is_reproducible(distribution):
    trace = generate_trace(distribution=distribution, shared_experts=2)
    assert trace == generate_trace(distribution=distribution, shared_experts=2)
    assert sum(summarize_routing(trace).routed_histogram) == 64
    assert trace.trace_kind == "synthetic"
    assert all(math.isclose(sum(t.weights), 1) for t in trace.tokens)


def test_hand_calculated_workload():
    trace = RoutingTrace.from_json(GOLDEN.read_text())
    config = WorkloadConfig(hidden_size=3, intermediate_size=5,
                            block_m=2, block_n=4, block_k=2, num_sms=2)
    work = compile_workload(trace, config)
    # Routed M=2,2,3,1 and shared M=4,4. Padded M sum=18.
    assert [g["m"] for g in work["groups"]] == [2, 2, 3, 1, 4, 4]
    assert work["totals"]["useful_flops"] == 6 * 16 * 3 * 5
    assert work["totals"]["padded_flops"] == 2 * (2 * 18 * 8 * 4) + 2 * 18 * 4 * 6
    assert [p["output_tiles"] for p in work["projections"]] == [18, 18, 9]
    assert [p["estimated_waves"] for p in work["projections"]] == [9, 9, 5]
    assert work["projections"][2]["last_wave_occupancy"] == .5
    assert work == compile_workload(RoutingTrace.from_json(trace.to_json()), config)


def test_inactive_experts_have_no_tiles_or_dispatch():
    work = compile_workload(generate_trace(tokens=3, distribution="skewed"), WorkloadConfig())
    for group in work["groups"][2:]:
        assert group["token_rows"] == []
    for projection in work["projections"]:
        assert all(g["output_tiles"] == g["padded_flops"] == 0 for g in projection["gemms"][2:])


@pytest.mark.parametrize("seed", range(20))
def test_randomized_dispatch_replay(seed):
    rng = random.Random(seed)
    experts = rng.randint(1, 8)
    trace = generate_trace(tokens=rng.randint(1, 10), experts=experts,
                           top_k=rng.randint(1, experts), shared_experts=rng.randint(0, 2),
                           distribution=rng.choice(["balanced", "random", "skewed"]), seed=seed)
    # Token identifiers are provenance, not array row offsets.
    trace = replace(trace, tokens=tuple(replace(t, token_index=100 + i * 3) for i, t in enumerate(trace.tokens)))
    config = WorkloadConfig(hidden_size=rng.randint(1, 5), intermediate_size=rng.randint(1, 7))
    inputs = make_inputs(trace, config, seed)
    actual = grouped_replay(trace, compile_workload(trace, config), inputs)
    assert compare_outputs(reference_replay(trace, inputs), actual)["passed"]


def test_one_dimensional_expert_has_hand_calculated_output():
    trace = route_scores([[1.0]], top_k=1, shared_experts=1)
    inputs = {"x": [[2.]], "experts": [{"gate": [[1.]], "up": [[3.]], "down": [[4.]]}] * 2}
    expected = 2 * (2 / (1 + math.exp(-2))) * 6 * 4
    assert reference_replay(trace, inputs)[0][0] == pytest.approx(expected)


def test_replay_detects_compiler_dispatch_regression():
    trace = generate_trace(tokens=3, experts=2)
    config = WorkloadConfig(hidden_size=2, intermediate_size=3)
    work, inputs = compile_workload(trace, config), make_inputs(trace, config, 1)
    work["groups"][0]["token_rows"] = []
    assert not compare_outputs(reference_replay(trace, inputs), grouped_replay(trace, work, inputs))["passed"]


def test_benchmark_records_raw_cpu_evidence():
    trace = generate_trace(tokens=2, experts=2)
    config = WorkloadConfig(hidden_size=2, intermediate_size=3)
    result = benchmark(trace, config, repeats=3, warmup=0)
    assert result["artifact_kind"] == "measured_cpu_replay"
    assert result["correctness"]["passed"]
    assert result["workload_sha256"] == artifact_hash(result["workload"])
    assert all(len(b["samples_ms"]) == 3 for b in result["backends"].values())
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("field", WorkloadConfig.__dataclass_fields__)
@pytest.mark.parametrize("value", [0, -1, True, 1.2])
def test_config_rejects_invalid_dimensions(field, value):
    with pytest.raises(ValueError):
        WorkloadConfig(**{field: value})


def test_parser_rejects_duplicate_fields():
    with pytest.raises(TraceValidationError, match="duplicate JSON"):
        RoutingTrace.from_json('{"tokens": [], "tokens": []}')


def test_normalization_contract_and_boolean_index():
    trace = generate_trace(tokens=1)
    with pytest.raises(TraceValidationError, match="sum to one"):
        replace(trace, tokens=(replace(trace.tokens[0], weights=(.1, .1)),))
    with pytest.raises(TraceValidationError, match="integer"):
        TokenRoute(True, (0,), (1.0,))


def test_cli_round_trip_and_errors(tmp_path):
    def cli(*args):
        return subprocess.run([sys.executable, "-m", "moescope", *map(str, args)], capture_output=True, text=True)
    trace = tmp_path / "trace.json"
    assert cli("generate", "--output", trace).returncode == 0
    assert json.loads(cli("validate", trace).stdout)["valid"]
    assert json.loads(cli("compile", trace).stdout)["artifact_kind"] == "compiled_workload"
    trace.write_text('{"bad": true}')
    failure = cli("validate", trace)
    assert failure.returncode == 2 and "Traceback" not in failure.stderr
