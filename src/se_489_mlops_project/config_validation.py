"""Configuration schema validation for Hydra/OmegaConf configs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig, OmegaConf


class ConfigValidationError(ValueError):
    """Raised when the training configuration fails validation."""


@dataclass
class _FieldSpec:
    path: str
    required: bool = True
    min_val: float | None = None
    max_val: float | None = None
    allowed: list[Any] = field(default_factory=list)
    type_: type | None = None


_FIELD_SPECS = [
    _FieldSpec("experiment.name",        type_=str),
    _FieldSpec("data.processed_dir",     type_=str),
    _FieldSpec("data.processed_file",    type_=str),
    _FieldSpec("data.target",            type_=str),
    _FieldSpec("model.type",             allowed=["random_forest"]),
    _FieldSpec("model.n_estimators",     min_val=1, max_val=10_000, type_=int),
    _FieldSpec("model.random_state",     min_val=0, type_=int),
    _FieldSpec("training.test_size",     min_val=0.05, max_val=0.5, type_=float),
    _FieldSpec("training.random_state",  min_val=0, type_=int),
]


def _get_nested(cfg: DictConfig, dotpath: str) -> Any:
    node: Any = cfg
    for part in dotpath.split("."):
        if not OmegaConf.is_config(node) or part not in node:
            raise KeyError(dotpath)
        node = node[part]
    return node


def validate_config(cfg: DictConfig) -> None:
    """Validate a Hydra DictConfig before training starts.

    Raises ConfigValidationError listing ALL problems at once.
    """
    errors: list[str] = []

    for spec in _FIELD_SPECS:
        try:
            value = _get_nested(cfg, spec.path)
        except KeyError:
            if spec.required:
                errors.append(f"  [{spec.path}] required field is missing")
            continue

        if spec.type_ is not None and not isinstance(value, spec.type_):
            errors.append(
                f"  [{spec.path}] expected {spec.type_.__name__}, "
                f"got {type(value).__name__} ({value!r})"
            )
            continue

        if spec.min_val is not None and value < spec.min_val:
            errors.append(f"  [{spec.path}] {value!r} is below minimum {spec.min_val}")

        if spec.max_val is not None and value > spec.max_val:
            errors.append(f"  [{spec.path}] {value!r} exceeds maximum {spec.max_val}")

        if spec.allowed and value not in spec.allowed:
            errors.append(
                f"  [{spec.path}] {value!r} not in allowed set {spec.allowed}"
            )

    if errors:
        raise ConfigValidationError(
            "Configuration validation failed:\n" + "\n".join(errors)
        )
