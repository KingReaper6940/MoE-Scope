"""Versioned, dependency-free routing trace schema."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Mapping, Self

TRACE_SCHEMA_VERSION = "1.0.0"

TraceKind = Literal["captured", "synthetic"]
InferencePhase = Literal["prefill", "decode"]


class TraceValidationError(ValueError):
    """Raised when a routing trace violates a schema invariant."""


def _require_exact_keys(
    payload: Mapping[str, Any],
    *,
    expected: set[str],
    location: str,
) -> None:
    missing = expected - payload.keys()
    unknown = payload.keys() - expected

    if missing:
        names = ", ".join(sorted(missing))
        raise TraceValidationError(f"{location} is missing fields: {names}")
    if unknown:
        names = ", ".join(sorted(unknown))
        raise TraceValidationError(f"{location} has unknown fields: {names}")


def _require_int(value: Any, *, location: str) -> int:
    if type(value) is not int:
        raise TraceValidationError(f"{location} must be an integer")
    return value


def _require_string(value: Any, *, location: str) -> str:
    if not isinstance(value, str):
        raise TraceValidationError(f"{location} must be a string")
    return value


@dataclass(frozen=True, slots=True)
class RoutingRule:
    """How a model turns router outputs into routed expert assignments."""

    score_function: str
    selection_method: str
    weight_normalization: str

    _FIELDS: ClassVar[set[str]] = {
        "score_function",
        "selection_method",
        "weight_normalization",
    }

    def __post_init__(self) -> None:
        for field_name in self._FIELDS:
            value = _require_string(getattr(self, field_name), location=f"routing_rule.{field_name}")
            if not value.strip():
                raise TraceValidationError(
                    f"routing_rule.{field_name} must not be empty"
                )

    def to_dict(self) -> dict[str, str]:
        return {
            "score_function": self.score_function,
            "selection_method": self.selection_method,
            "weight_normalization": self.weight_normalization,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        _require_exact_keys(
            payload,
            expected=cls._FIELDS,
            location="routing_rule",
        )
        return cls(
            score_function=_require_string(
                payload["score_function"],
                location="routing_rule.score_function",
            ),
            selection_method=_require_string(
                payload["selection_method"],
                location="routing_rule.selection_method",
            ),
            weight_normalization=_require_string(
                payload["weight_normalization"],
                location="routing_rule.weight_normalization",
            ),
        )


@dataclass(frozen=True, slots=True)
class TokenRoute:
    """Routed expert choices for one token position."""

    token_index: int
    expert_ids: tuple[int, ...]
    weights: tuple[float, ...]

    _FIELDS: ClassVar[set[str]] = {
        "token_index",
        "expert_ids",
        "weights",
    }

    def __post_init__(self) -> None:
        _require_int(self.token_index, location="token_index")
        if self.token_index < 0:
            raise TraceValidationError("token_index must be non-negative")
        if len(self.expert_ids) != len(self.weights):
            raise TraceValidationError(
                f"token {self.token_index} has {len(self.expert_ids)} expert IDs "
                f"but {len(self.weights)} weights"
            )
        if len(set(self.expert_ids)) != len(self.expert_ids):
            raise TraceValidationError(
                f"token {self.token_index} contains duplicate routed expert IDs"
            )

        for expert_id in self.expert_ids:
            if type(expert_id) is not int:
                raise TraceValidationError(
                    f"token {self.token_index} expert IDs must be integers"
                )

        for weight in self.weights:
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise TraceValidationError("routing weights must be numbers")
            if not math.isfinite(weight):
                raise TraceValidationError(
                    f"token {self.token_index} routing weights must be finite"
                )
            if weight < 0:
                raise TraceValidationError(
                    f"token {self.token_index} routing weights must be non-negative"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "token_index": self.token_index,
            "expert_ids": list(self.expert_ids),
            "weights": list(self.weights),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        _require_exact_keys(
            payload,
            expected=cls._FIELDS,
            location="token route",
        )

        raw_expert_ids = payload["expert_ids"]
        raw_weights = payload["weights"]
        if not isinstance(raw_expert_ids, list):
            raise TraceValidationError("token route expert_ids must be a list")
        if not isinstance(raw_weights, list):
            raise TraceValidationError("token route weights must be a list")

        token_index = _require_int(
            payload["token_index"],
            location="token route token_index",
        )
        expert_ids = tuple(
            _require_int(value, location=f"token {token_index} expert ID")
            for value in raw_expert_ids
        )

        weights: list[float] = []
        for value in raw_weights:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TraceValidationError(
                    f"token {token_index} routing weights must be numbers"
                )
            weights.append(float(value))

        return cls(
            token_index=token_index,
            expert_ids=expert_ids,
            weights=tuple(weights),
        )


@dataclass(frozen=True, slots=True)
class RoutingTrace:
    """A normalized routing event that is independent of benchmark results."""

    schema_version: str
    trace_kind: TraceKind
    model_name: str
    model_revision: str
    phase: InferencePhase
    layer_index: int
    step_index: int
    num_routed_experts: int
    num_shared_experts: int
    routed_experts_per_token: int
    routing_rule: RoutingRule
    tokens: tuple[TokenRoute, ...]

    _FIELDS: ClassVar[set[str]] = {
        "schema_version",
        "trace_kind",
        "model_name",
        "model_revision",
        "phase",
        "layer_index",
        "step_index",
        "num_routed_experts",
        "num_shared_experts",
        "routed_experts_per_token",
        "routing_rule",
        "tokens",
    }

    def __post_init__(self) -> None:
        for name in ("layer_index", "step_index", "num_routed_experts",
                     "num_shared_experts", "routed_experts_per_token"):
            _require_int(getattr(self, name), location=name)
        for name in ("model_name", "model_revision"):
            _require_string(getattr(self, name), location=name)
        if self.schema_version != TRACE_SCHEMA_VERSION:
            raise TraceValidationError(
                f"unsupported trace schema version {self.schema_version!r}; "
                f"expected {TRACE_SCHEMA_VERSION!r}"
            )
        if self.trace_kind not in ("captured", "synthetic"):
            raise TraceValidationError(
                "trace_kind must be 'captured' or 'synthetic'"
            )
        if self.phase not in ("prefill", "decode"):
            raise TraceValidationError("phase must be 'prefill' or 'decode'")
        if not self.model_name.strip():
            raise TraceValidationError("model_name must not be empty")
        if not self.model_revision.strip():
            raise TraceValidationError("model_revision must not be empty")
        if self.layer_index < 0:
            raise TraceValidationError("layer_index must be non-negative")
        if self.step_index < 0:
            raise TraceValidationError("step_index must be non-negative")
        if self.num_routed_experts <= 0:
            raise TraceValidationError("num_routed_experts must be positive")
        if self.num_shared_experts < 0:
            raise TraceValidationError("num_shared_experts must be non-negative")
        if not 0 < self.routed_experts_per_token <= self.num_routed_experts:
            raise TraceValidationError(
                "routed_experts_per_token must be between 1 and "
                "num_routed_experts"
            )
        if not self.tokens:
            raise TraceValidationError("a routing trace must contain at least one token")

        token_indices = [token.token_index for token in self.tokens]
        if len(set(token_indices)) != len(token_indices):
            raise TraceValidationError("token indices must be unique")

        for token in self.tokens:
            if self.routing_rule.weight_normalization == "sum_to_one" and not math.isclose(
                math.fsum(token.weights), 1.0, rel_tol=1e-6, abs_tol=1e-6
            ):
                raise TraceValidationError(f"token {token.token_index} weights must sum to one")
            if len(token.expert_ids) != self.routed_experts_per_token:
                raise TraceValidationError(
                    f"token {token.token_index} must select exactly "
                    f"{self.routed_experts_per_token} routed experts"
                )
            for expert_id in token.expert_ids:
                if not 0 <= expert_id < self.num_routed_experts:
                    raise TraceValidationError(
                        f"token {token.token_index} routed expert ID {expert_id} "
                        f"is outside [0, {self.num_routed_experts})"
                    )

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)

    @property
    def routed_assignment_count(self) -> int:
        return self.num_tokens * self.routed_experts_per_token

    @property
    def shared_assignment_count(self) -> int:
        return self.num_tokens * self.num_shared_experts

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "trace_kind": self.trace_kind,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "phase": self.phase,
            "layer_index": self.layer_index,
            "step_index": self.step_index,
            "num_routed_experts": self.num_routed_experts,
            "num_shared_experts": self.num_shared_experts,
            "routed_experts_per_token": self.routed_experts_per_token,
            "routing_rule": self.routing_rule.to_dict(),
            "tokens": [token.to_dict() for token in self.tokens],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        _require_exact_keys(
            payload,
            expected=cls._FIELDS,
            location="routing trace",
        )

        raw_rule = payload["routing_rule"]
        raw_tokens = payload["tokens"]
        if not isinstance(raw_rule, Mapping):
            raise TraceValidationError("routing_rule must be an object")
        if not isinstance(raw_tokens, list):
            raise TraceValidationError("tokens must be a list")

        return cls(
            schema_version=_require_string(
                payload["schema_version"],
                location="schema_version",
            ),
            trace_kind=_require_string(
                payload["trace_kind"],
                location="trace_kind",
            ),
            model_name=_require_string(
                payload["model_name"],
                location="model_name",
            ),
            model_revision=_require_string(
                payload["model_revision"],
                location="model_revision",
            ),
            phase=_require_string(payload["phase"], location="phase"),
            layer_index=_require_int(
                payload["layer_index"],
                location="layer_index",
            ),
            step_index=_require_int(
                payload["step_index"],
                location="step_index",
            ),
            num_routed_experts=_require_int(
                payload["num_routed_experts"],
                location="num_routed_experts",
            ),
            num_shared_experts=_require_int(
                payload["num_shared_experts"],
                location="num_shared_experts",
            ),
            routed_experts_per_token=_require_int(
                payload["routed_experts_per_token"],
                location="routed_experts_per_token",
            ),
            routing_rule=RoutingRule.from_dict(raw_rule),
            tokens=tuple(
                TokenRoute.from_dict(token)
                if isinstance(token, Mapping)
                else _raise_non_object_token(index)
                for index, token in enumerate(raw_tokens)
            ),
        )

    @classmethod
    def from_json(cls, value: str) -> Self:
        def unique_object(pairs):
            result = {}
            for key, item in pairs:
                if key in result:
                    raise TraceValidationError(f"duplicate JSON field: {key}")
                result[key] = item
            return result
        try:
            payload = json.loads(value, object_pairs_hook=unique_object)
        except json.JSONDecodeError as error:
            raise TraceValidationError(f"invalid trace JSON: {error.msg}") from error

        if not isinstance(payload, Mapping):
            raise TraceValidationError("routing trace JSON must contain an object")
        return cls.from_dict(payload)


def _raise_non_object_token(index: int) -> Any:
    raise TraceValidationError(f"token at index {index} must be an object")
