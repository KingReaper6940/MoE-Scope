"""Reproducible A100 grouped-GEMM experiment; run after capture.py."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess

import torch
import triton

from moescope.routing import generate_trace
from moescope.trace import RoutingTrace, summarize_routing
from moescope.triton_backend import PreparedGroup
from moescope.replay import _source_provenance
from moescope.workloads import WorkloadConfig, artifact_hash, compile_workload

CONFIGS = [(16, 64, 32), (32, 64, 32), (64, 64, 32)]


def measure(run, *, warmup=10, repeats=30, inner=10):
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    # Capture 10 invocations so host launch gaps do not dominate tiny GEMMs.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            run()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / inner)
    return {"samples_ms": samples, "median_ms": statistics.median(samples),
            "p95_ms": sorted(samples)[math.ceil(.95 * repeats) - 1],
            "min_ms": min(samples), "max_ms": max(samples),
            "stdev_ms": statistics.stdev(samples)}


def validate(group, run):
    reference = [(a.float() @ b.float()).half() for a, b in zip(group.a, group.b)]
    actual = run()
    max_error = 0.
    for expected, observed in zip(reference, actual):
        torch.testing.assert_close(observed, expected, atol=.002, rtol=.01)
        if expected.numel():
            max_error = max(max_error, float((observed - expected).abs().max()))
    return {"passed": True, "max_abs_error": max_error, "atol": .002, "rtol": .01,
            "reference": "float32 torch.mm cast to float16; TF32 disabled"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, default=Path("artifacts/capture"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gpu"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "traces").mkdir(exist_ok=True)
    torch.manual_seed(20260912)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    properties = torch.cuda.get_device_properties(0)
    environment = {"gpu": properties.name, "vram_bytes": properties.total_memory,
                   "num_sms": properties.multi_processor_count,
                   "compute_capability": list(torch.cuda.get_device_capability()),
                   "torch": torch.__version__, "triton": triton.__version__,
                   "cuda": torch.version.cuda, "python": platform.python_version(),
                   "platform": platform.platform(),
                   "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
                   "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
                   "image_digest": "sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35"}
    # Irregular and empty groups exercise masking, including partial K and N.
    edge_checks = []
    for seed in range(12):
        torch.manual_seed(seed)
        shapes = [(0, 17, 19), (1 + seed, 31 + seed, 33), (33, 65, 47)]
        a = [torch.randn(m, k, device="cuda", dtype=torch.float16) * .1 for m, n, k in shapes]
        b = [torch.randn(k, n, device="cuda", dtype=torch.float16) * .1 for m, n, k in shapes]
        group = PreparedGroup(a, b)
        for bm, bn, bk in CONFIGS:
            check = validate(group, lambda: group.triton(block_m=bm, block_n=bn, block_k=bk))
            edge_checks.append({"seed": seed, "shapes": shapes, "config": [bm, bn, bk], **check})
    print(f"Passed {len(edge_checks)} irregular GPU correctness cases", flush=True)
    traces = []
    for tokens in (1, 8, 32, 128, 512, 2048):
        for top_k in (2, 4, 8):
            for distribution in ("balanced", "random", "skewed"):
                name = f"synthetic-{distribution}-t{tokens}-k{top_k}"
                trace = generate_trace(tokens=tokens, experts=64, top_k=top_k,
                                       distribution=distribution, seed=20260912)
                traces.append((name, trace, 256, 512, distribution))
    manifest = json.loads((args.capture / "manifest.json").read_text())
    for event in manifest["events"]:
        if event["layer"] in (0, 7, 15) and event["step"] in (0, 1):
            trace = RoutingTrace.from_json((args.capture / event["file"]).read_text())
            traces.append((event["file"].removesuffix(".json"), trace,
                           manifest["hidden_size"], manifest["intermediate_size"], "captured"))
    cases = []
    for index, (name, trace, hidden, intermediate, distribution) in enumerate(traces):
        torch.manual_seed(20260912 + index)
        histogram = summarize_routing(trace).routed_histogram
        (args.output / "traces" / f"{name}.json").write_text(trace.to_json() + "\n")
        # Replay ONE projection with random tensor values, preserving real M/N/K.
        # Routed trace weights and shared experts are not part of this GEMM experiment.
        a = [torch.randn(m, hidden, device="cuda", dtype=torch.float16) * .1 for m in histogram if m]
        b = [torch.randn(hidden, intermediate, device="cuda", dtype=torch.float16) * .1 for m in histogram if m]
        group = PreparedGroup(a, b)
        functions = {"torch_loop": group.torch}
        for bm, bn, bk in CONFIGS:
            functions[f"triton_m{bm}_n{bn}_k{bk}"] = lambda bm=bm, bn=bn, bk=bk: group.triton(block_m=bm, block_n=bn, block_k=bk)
        correctness = {key: validate(group, run) for key, run in functions.items()}
        # Rotate order by case, recording it for inspection.
        order = list(functions)
        shift = index % len(order)
        order = order[shift:] + order[:shift]
        timing = {key: measure(functions[key]) for key in order}
        workloads = {}
        for bm, bn, bk in CONFIGS:
            config = WorkloadConfig(hidden, intermediate, bm, bn, bk, properties.multi_processor_count)
            work = compile_workload(trace, config)
            projection = work["projections"][0]
            useful = sum(g["useful_flops"] for g in projection["gemms"])
            padded = sum(g["padded_flops"] for g in projection["gemms"])
            workloads[f"triton_m{bm}_n{bn}_k{bk}"] = {
                "workload_sha256": artifact_hash(work), "config": work["config"],
                "output_tiles": projection["output_tiles"],
                "estimated_waves": projection["estimated_waves"],
                "padding_fraction": 1 - useful / padded,
                "useful_flops": useful, "padded_flops": padded}
        case = {"id": name, "trace_kind": trace.trace_kind, "distribution": distribution,
                "trace_file": f"traces/{name}.json", "trace_sha256": artifact_hash(trace.to_dict()),
                "tokens": trace.num_tokens, "experts": trace.num_routed_experts,
                "top_k": trace.routed_experts_per_token, "histogram": histogram,
                "active_experts": sum(m > 0 for m in histogram),
                "hidden_size": hidden, "intermediate_size": intermediate,
                "seed": 20260912 + index, "backend_order": order,
                "correctness": correctness, "timings": timing, "workloads": workloads}
        cases.append(case)
        print(f"{index + 1}/{len(traces)} {name}: torch={timing['torch_loop']['median_ms']:.4f} ms", flush=True)
        # Incremental artifact survives a later failure, marked incomplete until final.
        report = {"schema_version": "1.0.0", "artifact_kind": "measured_gpu_grouped_gemm",
                  "complete": False, "created_at": datetime.now(timezone.utc).isoformat(),
                  "environment": environment, "provenance": _source_provenance(),
                  "protocol": {"seed": 20260912, "dtype": "float16", "accumulator": "float32",
                               "warmup": 10, "graph_warmup": 3, "repeats": 30, "inner": 10,
                               "timer": "CUDA events around CUDA Graph replay, divided by inner",
                               "scope": "one grouped gate/up-shaped projection, prepared tensors",
                               "excluded": ["model forward", "routing", "gather/scatter", "pointer-table creation", "allocation", "JIT compilation"],
                               "tensor_values": "seeded random; captured traces preserve routing histograms and model dimensions, not learned weights or activations",
                               "limits": ["one rented A100", "one measurement session", "no clock locking", "six authored prompts", "no end-to-end MoE speedup claim", "winner is in-sample, not a trained selector"]},
                  "edge_checks": edge_checks, "cases": cases}
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        if name == "synthetic-balanced-t128-k2":
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                with torch.profiler.record_function("torch_loop_prepared"):
                    group.torch()
                with torch.profiler.record_function("triton_prepared"):
                    group.triton()
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(args.output / "profile.json"))
            (args.output / "profile.txt").write_text(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
        del group, a, b
    report["complete"] = True
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "environment.txt").write_text(subprocess.check_output(["python", "-m", "pip", "freeze"], text=True))
    print("Benchmark complete", flush=True)


if __name__ == "__main__":
    main()
