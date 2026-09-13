"""Compile routes into an explicit, analytical three-projection SwiGLU workload."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

from .trace import RoutingTrace, summarize_routing


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def artifact_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


@dataclass(frozen=True)
class WorkloadConfig:
    hidden_size: int = 64
    intermediate_size: int = 128
    block_m: int = 16
    block_n: int = 32
    block_k: int = 32
    num_sms: int = 80
    ctas_per_sm: int = 1

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


def compile_workload(trace: RoutingTrace, config: WorkloadConfig) -> dict:
    """No GPU is queried: waves assume one CTA per output tile and fixed capacity.

    All routed/shared experts use equal dimensions and three separate launches
    (gate, up, down). Shared expert outputs are summed without a learned gate.
    These are explicit replay assumptions, not universal MoE model semantics.
    """
    summary = summarize_routing(trace)
    groups = []
    for kind, counts in (("routed", summary.routed_histogram),
                         ("shared", [trace.num_tokens] * trace.num_shared_experts)):
        for expert, count in enumerate(counts):
            rows = [i for i, token in enumerate(trace.tokens)
                    if kind == "shared" or expert in token.expert_ids]
            groups.append({"kind": kind, "expert_id": expert, "m": count,
                           "token_rows": rows})
    projections = []
    for name, n, k in (("gate", config.intermediate_size, config.hidden_size),
                       ("up", config.intermediate_size, config.hidden_size),
                       ("down", config.hidden_size, config.intermediate_size)):
        gemms = []
        for group in groups:
            m = group["m"]
            pm = ceil_div(m, config.block_m) * config.block_m
            pn = ceil_div(n, config.block_n) * config.block_n
            pk = ceil_div(k, config.block_k) * config.block_k
            tiles = ceil_div(m, config.block_m) * ceil_div(n, config.block_n)
            gemms.append({"kind": group["kind"], "expert_id": group["expert_id"],
                          "m": m, "n": n, "k": k, "output_tiles": tiles,
                          "k_iterations": ceil_div(k, config.block_k),
                          "useful_flops": 2 * m * n * k,
                          "padded_flops": 2 * pm * pn * pk})
        tiles = sum(g["output_tiles"] for g in gemms)
        capacity = config.num_sms * config.ctas_per_sm
        waves = ceil_div(tiles, capacity)
        projections.append({"name": name, "gemms": gemms, "output_tiles": tiles,
                            "estimated_waves": waves,
                            "last_wave_occupancy": (tiles - (waves - 1) * capacity) / capacity
                            if waves else 0.0})
    useful = sum(g["useful_flops"] for p in projections for g in p["gemms"])
    padded = sum(g["padded_flops"] for p in projections for g in p["gemms"])
    return {
        "schema_version": "1.0.0", "artifact_kind": "compiled_workload",
        "trace_sha256": artifact_hash(trace.to_dict()), "trace_kind": trace.trace_kind,
        "config": asdict(config), "architecture": "swiglu-three-projection",
        "assumptions": ["equal dimensions for routed and shared experts",
                        "shared outputs summed with weight one",
                        "one CTA per output tile, fixed CTA capacity",
                        "three separate grouped launches; no fusion",
                        "no dispatch, memory, occupancy, or latency model"],
        "routing": summary.to_dict(), "groups": groups, "projections": projections,
        "totals": {"useful_flops": useful, "padded_flops": padded,
                   "padding_fraction": 1 - useful / padded if padded else 0.0,
                   "output_tiles": sum(p["output_tiles"] for p in projections),
                   "estimated_waves": sum(p["estimated_waves"] for p in projections)},
    }
