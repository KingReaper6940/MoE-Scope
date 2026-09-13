# Phase 1: from router scores to a trustworthy trace

In Phase 1, I build the smallest routing trace I can trust.

The input is one Mixture-of-Experts routing event. The output is versioned JSON
that records which routed experts each token selected, the weights attached to
those selections, and how many shared experts ran alongside them.

I want to answer a few basic questions before touching GPU kernels:

- How many experts did each token select?
- How many tokens reached each routed expert?
- How much work came from always-active shared experts?
- Can I serialize the trace, load it again, and get the same answer?

Triton kernels and GPU benchmarks come later. Phase 1 defines the model event
that those experiments will replay.

## The routing event

An MoE layer contains several feed-forward networks called experts. A router
scores those experts for each token and selects a small subset.

For `N` tokens, `E` routed experts, and a top-k value of `K`, the core tensors
look like this:

| Value | Shape | Meaning |
|---|---:|---|
| Router scores | `[N, E]` | One score for every token and routed expert |
| Selected expert IDs | `[N, K]` | The routed experts chosen for each token |
| Routing weights | `[N, K]` | Each selected expert's contribution |
| Routed histogram | `[E]` | The number of assignments received by each expert |

The first invariant follows directly from those shapes:

```text
sum(routed histogram) = N × K
```

If the equality fails, the trace lost or invented an assignment.

## The four-token fixture

I use four tokens, four routed experts, and top-2 routing as the canonical
example:

| Token | E0 | E1 | E2 | E3 | Selected |
|---|---:|---:|---:|---:|---|
| T0 | 0.60 | 0.10 | 0.20 | 0.10 | E0, E2 |
| T1 | 0.05 | 0.70 | 0.20 | 0.05 | E1, E2 |
| T2 | 0.40 | 0.10 | 0.45 | 0.05 | E2, E0 |
| T3 | 0.10 | 0.30 | 0.20 | 0.40 | E3, E1 |

Flattening the selected IDs gives:

```text
[0, 2, 1, 2, 2, 0, 3, 1]
```

Counting each ID gives the routed histogram:

```text
[2, 2, 3, 1]
```

There are eight assignments because four tokens each select two routed
experts:

```text
4 × 2 = 8
2 + 2 + 3 + 1 = 8
```

This example is small enough to calculate on paper. That makes it useful as an
oracle for the code.

## Routed and shared experts

Routed experts run when the router selects them. Shared experts run for every
token because the model architecture says so.

With two shared experts, the fixture creates:

```text
routed calls = 4 tokens × 2 routed experts = 8
shared calls = 4 tokens × 2 shared experts = 8
total calls  = 16
```

I keep shared experts out of the selected expert IDs. Appending them would turn
a real top-2 decision into a made-up top-4 decision.

The trace therefore stores two different facts:

- per-token routed IDs describe what the router selected;
- `num_shared_experts` describes the always-active architecture.

The summary combines both when it calculates the full expert workload.

## How the code maps to the idea

I read the implementation in this order:

1. [`examples/traces/four-token-top2.json`](../examples/traces/four-token-top2.json)
   contains the canonical routing event.
2. [`src/moescope/trace/schema.py`](../src/moescope/trace/schema.py) defines the
   routing rule, token route, and complete trace.
3. [`src/moescope/trace/summary.py`](../src/moescope/trace/summary.py) derives
   routed and shared assignment counts.
4. [`tests/trace/test_golden_trace.py`](../tests/trace/test_golden_trace.py)
   checks the fixture against the hand-worked answer.
5. [`tests/trace/test_schema.py`](../tests/trace/test_schema.py) rejects invalid
   shapes, duplicate IDs, out-of-range IDs, non-finite weights, unknown fields,
   and unsupported schema versions.

The schema uses frozen Python dataclasses and has no machine-learning framework
dependency. A trace stays a portable data object instead of becoming a PyTorch
object.

## How to learn Phase 1

I would learn this phase in six passes.

### 1. Understand one MoE layer

Start with the difference between total parameters and active parameters. Then
trace one token through router scores, top-k selection, expert execution, and
the weighted output.

You should be able to explain why a model can own many experts while using only
a few for each token.

### 2. Learn the tensor shapes

Work with `N`, `E`, and `K` before using real model dimensions. Write down the
shape of the scores, selected IDs, weights, and histogram.

Check how each shape changes when you add tokens, experts, or routed selections.

### 3. Calculate the fixture by hand

Use the four score rows above. Select the two largest values in each row,
flatten the IDs, and count them.

Do this before running the example. The hand calculation gives you an answer
independent of the implementation.

### 4. Read the fixture before the schema

The JSON shows the data without Python abstractions. Once every field makes
sense, read the dataclasses and their validation rules.

Try to predict which of these edits should fail:

1. Change an expert ID from `3` to `4`.
2. Give one token a single expert ID.
3. Repeat the same expert ID for one token.
4. Add a `latency_ms` field.
5. Change the number of shared experts from two to zero.

The first four corrupt the schema. The fifth creates a different valid
workload.

### 5. Run the tests and summary

From the repository root:

```bash
uv sync --dev
uv run pytest
uv run python examples/summarize_trace.py
```

The summary should report:

```json
{
  "num_tokens": 4,
  "routed_histogram": [2, 2, 3, 1],
  "routed_assignment_count": 8,
  "shared_assignment_count": 8,
  "total_assignment_count": 16
}
```

### 6. Reproduce the fixture with PyTorch

Create a `4 × 4` score tensor, run `torch.topk(..., k=2)`, and count the
flattened IDs with `torch.bincount`.

The tensor result should match the JSON fixture exactly. After that boundary
works, the same adapter can capture router outputs from OLMoE without changing
the trace schema.

## What completes Phase 1

Phase 1 has one definition of done: a fresh checkout can produce or load a
trace, validate it, and reproduce its summary.

The complete phase includes:

- the hand-checkable fixture;
- a versioned schema with routed and shared expert semantics;
- deterministic validation and summaries;
- a PyTorch adapter that reproduces the fixture;
- one captured OLMoE routing event;
- a command-line path for validating and summarizing traces.

The first GPU question starts after this boundary. Phase 2 will translate an
expert histogram into grouped matrix-multiplication shapes, tiles, padding, and
possible execution waves.

## Reading

I use these sources for the concepts behind Phase 1:

1. [OLMoE: Accelerating the Science of Language Models](https://arxiv.org/abs/2409.02060)
   for the model architecture and routing behavior.
2. [Hugging Face's OLMoE implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/olmoe)
   for the path from router logits to expert execution.
3. [PyTorch `topk`](https://docs.pytorch.org/docs/stable/generated/torch.topk.html)
   and [`bincount`](https://docs.pytorch.org/docs/stable/generated/torch.bincount.html)
   for the tensor operations used in the small example.
4. [FlashInfer Trace](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace)
   for a related approach to serializing inference workloads.
5. [Triton's grouped GEMM tutorial](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html)
   for a preview of the GPU workload Phase 2 will study.
