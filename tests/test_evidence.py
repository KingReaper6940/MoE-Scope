"""Release artifacts must independently validate without requiring CUDA."""
import json
import math
from pathlib import Path
import statistics
import hashlib
import zipfile

import pytest

from moescope.trace import RoutingTrace, summarize_routing
from moescope.workloads import WorkloadConfig, artifact_hash, compile_workload

ROOT = Path(__file__).resolve().parents[1] / "examples/evidence"
pytestmark = pytest.mark.skipif(not (ROOT / "gpu/results.json").exists(), reason="GPU evidence not present")


def test_archived_source_matches_executed_source_manifest():
    manifest = json.loads((ROOT / "source-manifest.json").read_text())
    with zipfile.ZipFile(ROOT / "source.zip") as archive:
        for name, digest in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest


def test_capture_artifacts_match_manifest_and_histograms():
    manifest = json.loads((ROOT / "capture/manifest.json").read_text())
    assert len(manifest["events"]) == 480
    assert len(manifest["prompts"]) == 6
    assert len({e["file"] for e in manifest["events"]}) == 480
    assert all(c["passed"] for c in manifest["learned_block_checks"])
    for event in manifest["events"]:
        trace = RoutingTrace.from_json((ROOT / "capture" / event["file"]).read_text())
        assert artifact_hash(trace.to_dict()) == event["sha256"]
        assert trace.trace_kind == "captured"
        assert trace.model_revision == manifest["revision"]
        assert list(summarize_routing(trace).routed_histogram) == event["histogram"]
        assert event["dispatch_counts_verified"]


def test_gpu_artifacts_match_compiler_and_raw_statistics():
    report = json.loads((ROOT / "gpu/results.json").read_text())
    assert report["complete"] and len(report["cases"]) == 90
    assert len(report["edge_checks"]) == 36
    assert all(check["passed"] for check in report["edge_checks"])
    assert report["environment"]["gpu"] == "NVIDIA A100-SXM4-40GB"
    for case in report["cases"]:
        trace = RoutingTrace.from_json((ROOT / "gpu" / case["trace_file"]).read_text())
        assert artifact_hash(trace.to_dict()) == case["trace_sha256"]
        assert trace.trace_kind == case["trace_kind"]
        assert list(summarize_routing(trace).routed_histogram) == case["histogram"]
        assert len(case["correctness"]) == len(case["timings"]) == 4
        assert all(check["passed"] for check in case["correctness"].values())
        for name, record in case["workloads"].items():
            compiled = compile_workload(trace, WorkloadConfig(**record["config"]))
            assert artifact_hash(compiled) == record["workload_sha256"]
            assert compiled["projections"][0]["output_tiles"] == record["output_tiles"]
        for timing in case["timings"].values():
            samples = timing["samples_ms"]
            assert len(samples) == report["protocol"]["repeats"] == 30
            assert all(math.isfinite(s) and s > 0 for s in samples)
            assert statistics.median(samples) == timing["median_ms"]
            assert sorted(samples)[math.ceil(.95 * len(samples)) - 1] == timing["p95_ms"]
