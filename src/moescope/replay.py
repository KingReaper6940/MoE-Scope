"""Small CPU correctness lab. Python timings are not GPU performance evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time

from . import __version__
from .trace import RoutingTrace
from .workloads import WorkloadConfig, artifact_hash, compile_workload


def make_inputs(trace: RoutingTrace, config: WorkloadConfig, seed: int) -> dict:
    rng = random.Random(seed)
    def matrix(rows, cols):
        return [[rng.uniform(-0.5, 0.5) for _ in range(cols)] for _ in range(rows)]
    h, f = config.hidden_size, config.intermediate_size
    return {"x": matrix(trace.num_tokens, h), "experts": [
        {"gate": matrix(h, f), "up": matrix(h, f), "down": matrix(f, h)}
        for _ in range(trace.num_routed_experts + trace.num_shared_experts)]}


def _expert(x, weights):
    def mv(vector, matrix):
        return [math.fsum(v * matrix[i][j] for i, v in enumerate(vector))
                for j in range(len(matrix[0]))]
    gate, up = mv(x, weights["gate"]), mv(x, weights["up"])
    # Stable SiLU for either sign.
    activation = [(g / (1 + math.exp(-g)) if g >= 0
                   else g * math.exp(g) / (1 + math.exp(g))) * u
                  for g, u in zip(gate, up)]
    return mv(activation, weights["down"])


def reference_replay(trace: RoutingTrace, inputs: dict) -> list[list[float]]:
    """Token-major oracle, independent of the compiler's dispatch table."""
    output = []
    for row, token in enumerate(trace.tokens):
        selections = list(zip(token.expert_ids, token.weights)) + [
            (trace.num_routed_experts + e, 1.0) for e in range(trace.num_shared_experts)]
        pieces = [(weight, _expert(inputs["x"][row], inputs["experts"][expert]))
                  for expert, weight in selections]
        output.append([math.fsum(weight * y[j] for weight, y in pieces)
                       for j in range(len(inputs["x"][row]))])
    return output


def grouped_replay(trace: RoutingTrace, workload: dict, inputs: dict) -> list[list[float]]:
    """Replay compiler dispatch rows expert-first, then scatter weighted outputs."""
    if workload["trace_sha256"] != artifact_hash(trace.to_dict()):
        raise ValueError("workload was compiled from a different trace")
    output = [[0.0] * len(x) for x in inputs["x"]]
    for group in workload["groups"]:
        expert = group["expert_id"]
        shared = group["kind"] == "shared"
        weights = inputs["experts"][expert + (trace.num_routed_experts if shared else 0)]
        for row in group["token_rows"]:
            route = trace.tokens[row]
            weight = 1.0 if shared else route.weights[route.expert_ids.index(expert)]
            y = _expert(inputs["x"][row], weights)
            for j, value in enumerate(y):
                output[row][j] += weight * value
    return output


def compare_outputs(expected, actual, *, atol=1e-10, rtol=1e-9) -> dict:
    if len(expected) != len(actual) or any(len(a) != len(b) for a, b in zip(expected, actual)):
        return {"passed": False, "reason": "shape mismatch", "atol": atol, "rtol": rtol}
    errors = []
    passed = True
    for a, b in zip(expected, actual):
        for x, y in zip(a, b):
            if not math.isfinite(x) or not math.isfinite(y):
                return {"passed": False, "reason": "nonfinite output", "atol": atol, "rtol": rtol}
            error = abs(x - y)
            errors.append(error)
            passed &= error <= atol + rtol * abs(x)
    return {"passed": passed, "max_abs_error": max(errors, default=0.0),
            "atol": atol, "rtol": rtol}


def _source_provenance() -> dict:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(root), *args],
                                           stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git("status", "--porcelain")
    return {"package_version": __version__, "source_sha256": digest.hexdigest(),
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(status) if status is not None else None}


def benchmark(trace: RoutingTrace, config: WorkloadConfig, *, seed: int = 0,
              warmup: int = 2, repeats: int = 10) -> dict:
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    if type(repeats) is not int or repeats < 2:
        raise ValueError("repeats must be at least 2")
    workload = compile_workload(trace, config)
    inputs = make_inputs(trace, config, seed)
    reference = lambda: reference_replay(trace, inputs)
    grouped = lambda: grouped_replay(trace, workload, inputs)
    correctness = compare_outputs(reference(), grouped())
    if not correctness["passed"]:
        raise ValueError(f"replay correctness failed: {correctness}")
    backends = {"python_token_major": reference, "python_expert_major": grouped}
    for _ in range(warmup):
        for run in backends.values():
            run()
    samples = {name: [] for name in backends}
    # Alternate order to reduce a systematic first/second-run bias.
    for iteration in range(repeats):
        for name in list(backends)[::1 if iteration % 2 == 0 else -1]:
            start = time.perf_counter_ns()
            backends[name]()
            samples[name].append((time.perf_counter_ns() - start) / 1e6)
    return {
        "schema_version": "1.0.0", "artifact_kind": "measured_cpu_replay",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trace_sha256": workload["trace_sha256"], "trace_kind": trace.trace_kind,
        "workload_sha256": artifact_hash(workload), "inputs_sha256": artifact_hash(inputs),
        "workload": workload, "provenance": _source_provenance(),
        "environment": {"python": platform.python_version(), "os": platform.platform(),
                        "machine": platform.machine(), "processor": platform.processor(),
                        "device": "cpu", "arithmetic": "Python float (binary64)"},
        "protocol": {"seed": seed, "warmup": warmup, "repeats": repeats,
                     "clock": "perf_counter_ns", "order": "alternating",
                     "scope": "Python dispatch, expert arithmetic, scatter, and output allocation",
                     "excluded": "input generation, compilation, initial correctness check",
                     "limitations": "shared arithmetic oracle; no independent framework validation; no GPU inference claims"},
        "correctness": correctness,
        "backends": {name: {"samples_ms": values, "median_ms": statistics.median(values),
                            "min_ms": min(values), "max_ms": max(values),
                            "p95_ms": sorted(values)[math.ceil(0.95 * len(values)) - 1],
                            "stdev_ms": statistics.stdev(values)}
                     for name, values in samples.items()},
    }
