"""Build a compact browser dataset from versioned Python artifacts."""
import json
from pathlib import Path
import shutil

from moescope.trace import RoutingTrace
from moescope.routing import generate_trace
from moescope.workloads import WorkloadConfig, compile_workload

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "web/src/data"
PUBLIC = ROOT / "web/public/data"


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    traces = [{"id": "four-token-top2", "label": "Four tokens · hand-checked", "trace":
               RoutingTrace.from_json((ROOT / "examples/traces/four-token-top2.json").read_text()).to_dict()}]
    for distribution in ("balanced", "random", "skewed"):
        traces.append({"id": distribution, "label": f"Synthetic · {distribution}",
                       "trace": generate_trace(tokens=128, experts=16, top_k=2,
                                                shared_experts=2, distribution=distribution).to_dict()})
    capture = ROOT / "examples/evidence/capture"
    manifest = None
    if (capture / "manifest.json").exists():
        manifest = json.loads((capture / "manifest.json").read_text())
        for event in manifest["events"]:
            if event["layer"] in (0, 7, 15) and event["step"] in (0, 1):
                traces.append({"id": event["file"].removesuffix(".json"),
                               "label": f'OLMoE · {event["prompt_id"]} · L{event["layer"]} · {event["phase"]}',
                               "trace": json.loads((capture / event["file"]).read_text())})
        shutil.copy2(capture / "manifest.json", PUBLIC / "capture-manifest.json")
    checks = [{"trace": item["trace"], "workload": compile_workload(RoutingTrace.from_dict(item["trace"]), config)}
              for item in traces[:6]
              for config in (WorkloadConfig(), WorkloadConfig(3, 5, 2, 4, 2, 2))]
    (TARGET / "compiler-checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    (TARGET / "traces.json").write_text(json.dumps(traces, separators=(",", ":")) + "\n")
    result_path = ROOT / "examples/evidence/gpu/results.json"
    results = json.loads(result_path.read_text()) if result_path.exists() else None
    (TARGET / "results.json").write_text(json.dumps(results, separators=(",", ":")) + "\n")
    if results:
        shutil.copy2(result_path, PUBLIC / "gpu-results.json")
        existing = {t["id"] for t in traces}
        for case in results["cases"]:
            if case["id"] not in existing:
                traces.append({"id": case["id"], "label": case["id"],
                               "trace": json.loads((result_path.parent / case["trace_file"]).read_text()),
                               "hidden_size": case["hidden_size"], "intermediate_size": case["intermediate_size"]})
            else:
                entry = next(t for t in traces if t["id"] == case["id"])
                entry.update(hidden_size=case["hidden_size"], intermediate_size=case["intermediate_size"])
        (TARGET / "traces.json").write_text(json.dumps(traces, separators=(",", ":")) + "\n")
        shutil.copy2(result_path.parent / "profile.json", PUBLIC / "profile.json")
        shutil.copy2(result_path.parent / "profile.txt", PUBLIC / "profile.txt")
    summary = {"capture": {key: manifest[key] for key in (
        "created_at", "model", "revision", "num_layers", "num_experts", "top_k", "hidden_size",
        "intermediate_size", "dtype", "device", "learned_block_checks")} if manifest else None,
        "event_count": len(manifest["events"]) if manifest else 0,
        "prompt_count": len(manifest["prompts"]) if manifest else 0,
        "case_count": len(results["cases"]) if results else 0}
    (TARGET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (PUBLIC / "traces").mkdir(exist_ok=True)
    index = []
    for item in traces:
        (PUBLIC / "traces" / f'{item["id"]}.json').write_text(json.dumps(item["trace"], separators=(",", ":")) + "\n")
        index.append({key: value for key, value in item.items() if key != "trace"})
    (TARGET / "trace-index.json").write_text(json.dumps(index, separators=(",", ":")) + "\n")
    print(f'Exported {len(traces)} explorer traces; {summary["case_count"]} GPU cases')


if __name__ == "__main__":
    main()
