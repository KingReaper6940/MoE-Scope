# Recorded A100 evidence

This directory contains real artifacts from our September 12, 2026 local-date
experiment (September 13 in UTC). The GPU was NVIDIA A100-SXM4-40GB on Runpod.

- `capture/`: 480 normalized pretrained OLMoE routing events and a capture
  manifest with model revision, authored prompts, token IDs, dimensions,
  routing histograms, and learned-block validation.
- `gpu/results.json`: 90 prepared grouped-GEMM workloads, four backends,
  10,800 raw timing samples, correctness checks, configuration, and environment.
- `gpu/traces/`: the 90 exact benchmark routing inputs. Captured-routing cases
  preserve the model's histogram and dimensions but use seeded random tensor
  values for the GEMM replay. They do not replay learned activations or weights.
- `gpu/profile.json` and `gpu/profile.txt`: PyTorch CPU/CUDA profiling of the
  balanced 128-token/top-2 case. Instrumented durations are not the benchmark.
- `gpu/environment.txt`: the actual Python dependency snapshot.
- `source-manifest.json`: byte hashes of source, experiment scripts, and tests
  as executed on the pod, before later documentation and artifact-validation
  additions. The remote unpacked source had no Git checkout; its result bundle
  correctly records a null Git commit and a source-package hash.
- `logs/`: successful capture, benchmark, and GPU test output.

The recorded capture manifest's `seed: 0` was not explicitly set in the
archived capture script. The current script now sets it explicitly. Capture uses evaluation mode and greedy decode with no
stochastic sampling; it must not be interpreted as a seeded sampling experiment.
The GEMM benchmark explicitly sets and records its random seeds for each case.

The normalized trace hash uses canonical JSON with sorted keys, compact
separators, and UTF-8. It differs from hashing the pretty-printed file bytes.
The source manifest hashes raw source-file bytes, including line endings.
`source.zip` preserves the exact source payload executed in this run, including
the capture script before that seed-metadata fix. Its code is archival; current
installation and project status are described in the repository-root README.
`tests/test_evidence.py` verifies the released trace hashes, histograms,
compiler hashes, and statistics without CUDA.

The run completed successfully, all 570 capture/replay trace hashes were checked
locally, and the GPU pod was terminated. Provider billing records were still
empty immediately after teardown; the quoted compute rate was $1.00/hour,
with storage billed separately. Empty billing records do not mean the run was free.

See [the reproduction guide](../../docs/reproduce.md) for commands and the exact
measurement boundary. These results are not end-to-end inference speedups.
