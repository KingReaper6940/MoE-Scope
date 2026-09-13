# Reproduce MoEscope

Our local tools need Python 3.11 or newer. The website uses Node 22.13 or newer.
Our recorded GPU environment uses the pinned Runpod PyTorch image below.

## Local installation and demo

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e . pytest==8.4.2
python -m pytest
python -m moescope generate --tokens 128 --experts 64 --top-k 2 --distribution balanced --output trace.json
python -m moescope validate trace.json
python -m moescope summarize trace.json
python -m moescope compile trace.json --hidden-size 256 --intermediate-size 512 --block-m 32 --num-sms 108 --output workload.json
python -m moescope benchmark examples/traces/four-token-top2.json --output cpu-result.json
```

The CLI also installs as `moescope`. Core routing, compilation, and CPU replay
have no third-party runtime dependencies. CUDA imports are isolated in the
optional Triton backend. CPU timings include Python arithmetic and dispatch;
they do not estimate GPU performance.

## GPU capture and benchmark

The first experiment targets one **NVIDIA A100-SXM4-40GB**. Provision a CUDA-capable
Linux machine with `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`. Our run resolved
that tag to digest
`sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`.
The image supplies PyTorch 2.8.0+cu128 and Triton 3.4.0.

On that image, install into the existing interpreter so we retain its CUDA stack:

```bash
python -m pip install --break-system-packages -e . -r experiments/gpu/requirements.txt
python -m pytest -q
HF_HUB_ENABLE_HF_TRANSFER=0 HF_HOME=/workspace/hf-cache python experiments/gpu/capture.py
python experiments/gpu/benchmark.py
```

`HF_HUB_ENABLE_HF_TRANSFER=0` disables a template-default accelerator that may
be enabled without its package installed. The normal Hugging Face download
path works without adding it. The capture downloads the public OLMoE checkpoint;
allow enough disk for the weights and cache. The scripts run sequentially so
the model releases GPU memory before microbenchmark tensors are allocated.

For exact model reproduction, pass the revision from the released manifest:

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 HF_HOME=/workspace/hf-cache python experiments/gpu/capture.py --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5
```

Capture produces `artifacts/capture/manifest.json` and normalized trace files.
The manifest contains six authored prompts, tokenizer input IDs, generated IDs,
model revision, dimensions, dtype, environment, and adapter checks. Each prompt
records prefill and four decode events across all layers. The adapter recomputes
top-k from captured logits using the pinned model rule and independently checks
expert input counts. It also checks the first learned MoE block's output for
each prefill against explicit PyTorch gather/weight/scatter execution.

Benchmarking produces `artifacts/gpu/results.json`, trace inputs, a PyTorch
CPU/CUDA profile, and a dependency snapshot. A partially interrupted run is
marked `complete: false`. Do not publish it as a completed experiment.

## Measurement boundary

The GPU experiment measures one prepared gate/up-shaped grouped GEMM:

- 54 synthetic cases: 6 token counts × 3 top-k values × 3 distributions;
- 36 captured-routing cases: 6 prompts × 3 layers × 2 phases;
- PyTorch `torch.mm` loop and 3 fixed Triton configurations;
- float16 inputs/output with float32 accumulation;
- float32 PyTorch reference, cast to float16, TF32 disabled;
- 10 warmup calls, 3 warmup CUDA-graph replays;
- 30 event-timed samples, each containing 10 invocations in a CUDA graph;
- preallocated outputs and prepared pointer tables for both paths.

Input values are seeded random matrices. Captured-routing cases preserve model
histograms and dimensions, **not learned expert weights or activations**. Timings
exclude model inference, routing, gather/scatter, allocations, pointer-table
construction, and JIT compilation. CPU and GPU timing protocols differ and must
not be compared as if they measured the same operation.

The fastest tested configuration is selected from the same measurements. Ratios
are in-sample comparisons, not a held-out routing policy. The three-projection
workload compiler reports analytical padding and waves; the benchmark times
only one projection. The persistent kernel uses a fixed worker pool, so the
analytical wave estimate describes tile-work rounds, not measured occupancy.

## Website

```bash
python scripts/export_web.py
cd web
npm ci
npm test
npm run check
npm run build
npm run dev
```

From the repository root, `python scripts/check_site.py` checks the built site's
internal links, anchors, artifact downloads, and ten-chapter journal.
The browser loads generated, versioned data rather than inventing a second set
of benchmark numbers. `export_web.py` expects released evidence under
`examples/evidence/`; it also works before a GPU run, showing an explicit pending
state. CPU/compiler examples remain available in that state.

After copying completed GPU artifacts into `examples/evidence/`, regenerate the
web data, validate it, and rebuild. The new `MoE-Scope` repository publishes
through GitHub Pages using `.github/workflows/pages.yml`. See
[publishing](publish.md) for setup. Local preview also uses `/MoE-Scope/`,
so we verify the same paths that the deployed website uses.

## Limits and next experiments

We have one GPU family, one session, six authored prompts, and batch size one.
We do not lock clocks. The PyTorch profile documents launch structure; it does
not replace Nsight hardware-counter evidence. Further work should repeat runs,
test a held-out routing selector, measure end-to-end dispatch, use a broader
prompt set, and evaluate another GPU architecture.

After collecting and verifying the artifacts, terminate the rented pod. Stopping
it pauses GPU charges but retains billable storage.
