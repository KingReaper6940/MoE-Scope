"""Command line interface for inspectable routing experiments."""

import argparse
import json
from pathlib import Path
import sys

from .routing import generate_trace
from .trace import RoutingTrace, summarize_routing
from .workloads import WorkloadConfig, compile_workload
from .replay import benchmark


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="moescope", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="generate a labeled synthetic trace")
    for name, default in (("tokens", 32), ("experts", 8), ("top-k", 2), ("shared-experts", 0), ("seed", 0)):
        generate.add_argument(f"--{name}", type=int, default=default)
    generate.add_argument("--distribution", choices=["balanced", "random", "skewed"], default="balanced")
    generate.add_argument("--output", type=Path)
    for name in ("validate", "summarize", "compile", "benchmark"):
        command = commands.add_parser(name)
        command.add_argument("trace", type=Path)
        command.add_argument("--output", type=Path)
        if name in ("compile", "benchmark"):
            for field, default in WorkloadConfig().__dict__.items():
                if name == "benchmark" and field in ("hidden_size", "intermediate_size"):
                    default = 8 if field == "hidden_size" else 12
                command.add_argument("--" + field.replace("_", "-"), type=int, default=default)
        if name == "benchmark":
            command.add_argument("--warmup", type=int, default=2)
            command.add_argument("--repeats", type=int, default=10)
            command.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            result = generate_trace(tokens=args.tokens, experts=args.experts, top_k=args.top_k,
                                    shared_experts=args.shared_experts,
                                    distribution=args.distribution, seed=args.seed).to_dict()
        else:
            trace = RoutingTrace.from_json(args.trace.read_text(encoding="utf-8"))
            if args.command == "validate":
                result = {"valid": True, "schema_version": trace.schema_version,
                          "trace_kind": trace.trace_kind, "num_tokens": trace.num_tokens}
            elif args.command == "summarize":
                result = summarize_routing(trace).to_dict()
            else:
                config = WorkloadConfig(**{key: getattr(args, key) for key in WorkloadConfig.__dataclass_fields__})
                if args.command == "compile":
                    result = compile_workload(trace, config)
                else:
                    # Keep accidental model-sized runs out of this educational Python backend.
                    if trace.num_tokens * (trace.routed_experts_per_token + trace.num_shared_experts) * config.hidden_size * config.intermediate_size > 2_000_000:
                        raise ValueError("CPU demo is limited to 2 million assignment × hidden × intermediate elements; use smaller shapes")
                    result = benchmark(trace, config, warmup=args.warmup, repeats=args.repeats, seed=args.seed)
        content = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content, encoding="utf-8")
        else:
            print(content, end="")
    except (ValueError, OSError) as error:
        print(f"moescope: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
