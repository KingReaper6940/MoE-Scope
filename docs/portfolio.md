# Presenting the project

We can describe this release with concrete, reproducible engineering outcomes.
The project is useful because its measurements are connected to model routing,
not because every workload favors our kernel.

## Résumé-ready bullets

- Built MoEscope, an open-source Python/Triton/Astro lab connecting MoE model
  routing to validated traces, deterministic GPU workload compilation, and an
  interactive evidence explorer.
- Captured 480 routing events from a pinned pretrained OLMoE checkpoint and
  benchmarked 90 grouped-GEMM workloads on an NVIDIA A100, preserving 10,800
  latency samples with trace hashes, correctness checks, and environment data.
- Implemented a masked persistent Triton grouped-GEMM baseline, validated it
  against PyTorch, and documented both win and loss regions across three tile
  configurations and synthetic/model-derived workloads.

Use a speed ratio only with its scope: in the 128-token/top-2 balanced synthetic
case, prepared single-projection GEMM took 42.50 µs with the fastest tested
Triton configuration versus 261.79 µs for the PyTorch loop (6.16×). In the
matched concentrated case, Triton was slower (12.13 versus 10.30 µs). This is
not an end-to-end inference speedup or an independently validated selector.

## A two-minute demo

1. Open the explorer and select a captured OLMoE prefill event. Point out the
   model revision, token's selected experts, and actual routing weights.
2. Change the M tile. Explain why useful arithmetic stays fixed while padding
   and tile counts change. Explain the limits of the wave estimate.
3. Open the evidence page's matched 128-token controls. Explain why changing
   the histogram changes active matrix groups and PyTorch launch count.
4. Show a losing case alongside a winning one. Download the raw JSON and show
   the timing samples, tolerances, and excluded operations.
5. Run a CLI validation and the tests locally. Explain why no GPU is needed to
   inspect the released artifacts.

## Questions worth preparing for

- Why is a routing trace separate from a benchmark result?
- How do shared experts change the work count?
- What does the Triton kernel mask, and why are empty groups important?
- What is inside the CUDA timer? Why use graphs, and what do they exclude?
- Why can the PyTorch loop win on concentrated routing?
- What additional evidence would justify a routing-aware selector?
- Why do six authored prompts not establish production routing statistics?

## Attribution and chronology

We use the static scheduling design from Triton's official tutorial and a
pretrained OLMoE checkpoint. The initial contribution is an inspectable lab,
adapter, compiler, verification workflow, and evidence set. The journal labels
retrospectives and preserves actual measurement dates. These are collaborative
project outcomes; describe personal responsibilities consistently with the work
you can explain and reproduce.
