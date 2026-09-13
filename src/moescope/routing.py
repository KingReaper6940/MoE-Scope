"""Deterministic synthetic routing; no claim of model capture."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from .trace import RoutingRule, RoutingTrace, TokenRoute, TRACE_SCHEMA_VERSION


def route_scores(scores: Sequence[Sequence[float]], *, top_k: int,
                 shared_experts: int = 0, logits: bool = True,
                 normalize: bool = True, name: str = "moescope-scores",
                 revision: str = "v1") -> RoutingTrace:
    """Route supplied synthetic scores; exact ties prefer the lower expert ID.

    Precomputed scores must be nonnegative. Logits use a stable softmax over
    *all* experts before selection. Optional renormalization follows selection.
    This helper deliberately cannot label hand-supplied scores as captured.
    """
    if not scores or not scores[0]:
        raise ValueError("scores must contain at least one nonempty row")
    experts = len(scores[0])
    if type(top_k) is not int or not 1 <= top_k <= experts:
        raise ValueError("top_k must be between 1 and the expert count")
    tokens = []
    for index, row in enumerate(scores):
        if len(row) != experts:
            raise ValueError("score rows must have the same width")
        if any(isinstance(x, bool) or not isinstance(x, (int, float))
               or not math.isfinite(x) for x in row):
            raise ValueError("scores must be finite numbers")
        if logits:
            exps = [math.exp(x - max(row)) for x in row]
            total = math.fsum(exps)
            values = [x / total for x in exps]
        else:
            if min(row) < 0:
                raise ValueError("precomputed scores must be nonnegative")
            values = list(row)
        # Select from original scores: exponentiation can erase tiny differences.
        ids = tuple(sorted(range(experts), key=lambda e: (-row[e], e))[:top_k])
        weights = tuple(values[e] for e in ids)
        if normalize:
            total = math.fsum(weights)
            if total <= 0:
                raise ValueError("selected weights must have a positive sum")
            weights = tuple(w / total for w in weights)
        tokens.append(TokenRoute(index, ids, weights))
    return RoutingTrace(
        TRACE_SCHEMA_VERSION, "synthetic", name, revision, "prefill", 0, 0,
        experts, shared_experts, top_k,
        RoutingRule("softmax" if logits else "precomputed", "top_k",
                    "sum_to_one" if normalize else "not_normalized"), tuple(tokens),
    )


def generate_trace(*, tokens: int = 32, experts: int = 8, top_k: int = 2,
                   shared_experts: int = 0, distribution: str = "balanced",
                   seed: int = 0) -> RoutingTrace:
    """Generate balanced, random, or maximally concentrated synthetic routes."""
    if type(tokens) is not int or tokens < 1:
        raise ValueError("tokens must be a positive integer")
    if type(experts) is not int or experts < 1:
        raise ValueError("experts must be a positive integer")
    if type(top_k) is not int or not 1 <= top_k <= experts:
        raise ValueError("top_k must be between 1 and experts")
    if distribution not in ("balanced", "random", "skewed"):
        raise ValueError("distribution must be balanced, random, or skewed")
    rng = random.Random(seed)
    scores = []
    for token in range(tokens):
        row = [rng.random() for _ in range(experts)]
        if distribution != "random":
            start = token * top_k if distribution == "balanced" else 0
            for rank in range(top_k):
                row[(start + rank) % experts] = 2.0 + (top_k - rank) / top_k
        scores.append(row)
    return route_scores(scores, top_k=top_k, shared_experts=shared_experts,
                        name=f"moescope-synthetic-{distribution}",
                        revision=f"generator-v1-seed-{seed}")
