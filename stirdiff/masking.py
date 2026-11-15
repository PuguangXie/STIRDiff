"""Training-time observation-mask augmentation for six-vital windows.

The bank stores only finite/non-finite observation patterns from a caller's
training frame. It does not retain identifiers or clinical values. ``build_masks``
accepts observation masks and returns disjoint condition and target masks.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping

import numpy as np


VITAL_COLUMNS = ("heartrate", "sbp", "dbp", "resprate", "temperature", "spo2")
MASK_MODES = {"random", "historical", "time_block", "feature_block", "low_observation"}
_CLASSES = ("short_1", "short_2", "long")


def _scene_key(scene: Any) -> str:
    """Map equivalent numeric scene values to a common key."""
    if scene is None or (isinstance(scene, (float, np.floating)) and math.isnan(float(scene))):
        return "invalid:none"
    try:
        value = float(scene)
    except (TypeError, ValueError):
        return f"other:{str(scene)}"
    if math.isfinite(value) and value.is_integer() and 1 <= int(value) <= 5:
        return f"scene:{int(value)}"
    return f"other:{str(scene)}"


def _class_for_length(length: int) -> str:
    if length == 1:
        return "short_1"
    if length == 2:
        return "short_2"
    return "long"


def _as_array(frame: Any, name: str) -> np.ndarray:
    if hasattr(frame, "__getitem__"):
        return np.asarray(frame[name])
    raise TypeError("frame must provide named columns")


def _finite_column(frame: Any, name: str) -> np.ndarray:
    values = _as_array(frame, name)
    if values.dtype.kind in "biufc":
        return np.isfinite(values)
    try:
        return np.isfinite(values.astype(float))
    except (TypeError, ValueError):
        result = np.zeros(values.shape, dtype=bool)
        for index, value in enumerate(values):
            try:
                result[index] = value is not None and math.isfinite(float(value))
            except (TypeError, ValueError):
                result[index] = False
        return result


class MaskBank:
    """Bounded reservoir of training observation masks, grouped by scene."""

    def __init__(
        self,
        records: Mapping[tuple[str, str], list[np.ndarray]] | None = None,
        *,
        max_length: int = 256,
        per_scene: int = 512,
    ) -> None:
        self.max_length = int(max_length)
        self.per_scene = int(per_scene)
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.per_scene <= 0:
            raise ValueError("per_scene must be positive")
        self._records: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
        for key, values in (records or {}).items():
            scene, klass = key
            if klass not in _CLASSES:
                raise ValueError(f"unknown mask class: {klass}")
            self._records[(str(scene), klass)] = [
                np.asarray(value, dtype=bool).copy() for value in values
            ]

    @classmethod
    def fit(
        cls,
        frame: Any,
        seed: int = 2026,
        per_scene: int = 512,
        max_length: int = 256,
    ) -> "MaskBank":
        """Build a bounded bank from finite-value patterns in a training frame."""
        max_length = int(max_length)
        per_scene = int(per_scene)
        if max_length <= 0 or per_scene <= 0:
            raise ValueError("max_length and per_scene must be positive")
        required = ("hadm_id", "hospital_observationoffset", "__row_id", "scene")
        for name in required:
            try:
                _as_array(frame, name)
            except (KeyError, TypeError) as exc:
                raise ValueError(f"frame missing required column: {name}") from exc

        hadm = _as_array(frame, "hadm_id")
        scenes = _as_array(frame, "scene")
        offsets = _as_array(frame, "hospital_observationoffset")
        row_ids = _as_array(frame, "__row_id")
        n = len(scenes)
        if any(len(values) != n for values in (hadm, offsets, row_ids)):
            raise ValueError("frame columns have different lengths")
        try:
            offsets = np.asarray(offsets, dtype=float)
            row_ids = np.asarray(row_ids, dtype=np.int64)
        except (TypeError, ValueError) as exc:
            raise ValueError("sort columns must be numeric") from exc
        if not np.isfinite(offsets).all():
            raise ValueError("hospital_observationoffset must be finite")

        present = np.empty((6, n), dtype=bool)
        for index, name in enumerate(VITAL_COLUMNS):
            column = _finite_column(frame, name)
            if len(column) != n:
                raise ValueError("frame columns have different lengths")
            present[index] = column

        # Sort chronologically within each admission; row ID breaks offset ties.
        try:
            order = np.lexsort((row_ids, offsets, hadm.astype(str)))
        except (TypeError, ValueError) as exc:
            raise ValueError("hadm_id must be sortable") from exc
        hadm = hadm[order]
        scenes = scenes[order]
        present = present[:, order]

        rng = np.random.default_rng(seed)
        records: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
        seen: dict[tuple[str, str], int] = defaultdict(int)

        def add(mask: np.ndarray, scene_value: Any) -> None:
            length = int(mask.shape[0])
            if length == 0:
                return
            klass = _class_for_length(length)
            key = (_scene_key(scene_value), klass)
            seen[key] += 1
            candidate = np.asarray(mask, dtype=bool).copy()
            bucket = records[key]
            if len(bucket) < per_scene:
                bucket.append(candidate)
            else:
                slot = int(rng.integers(0, seen[key]))
                if slot < per_scene:
                    bucket[slot] = candidate

        if n:
            run_start = 0
            for row in range(1, n + 1):
                changed = (
                    row == n
                    or hadm[row] != hadm[run_start]
                    or _scene_key(scenes[row]) != _scene_key(scenes[run_start])
                )
                if not changed:
                    continue
                scene_value = scenes[run_start]
                for start in range(run_start, row, max_length):
                    stop = min(start + max_length, row)
                    add(present[:, start:stop].T, scene_value)
                run_start = row

        return cls(records, max_length=max_length, per_scene=per_scene)

    def _candidate_keys(self, scene: Any, length: int) -> list[tuple[str, str]]:
        scene_id = _scene_key(scene)
        preferred = _class_for_length(max(1, int(length)))
        exact = (scene_id, preferred)
        if self._records.get(exact):
            return [exact]
        return [
            key for key in self._records
            if key[0] == scene_id and self._records[key]
        ]

    def sample(
        self, scene: Any, length: int, rng: np.random.Generator
    ) -> np.ndarray | None:
        """Return a scene-matched boolean pattern, or ``None`` if unavailable."""
        length = int(length)
        if length <= 0:
            return np.zeros((0, 6), dtype=bool)
        keys = self._candidate_keys(scene, length)
        if not keys:
            return None
        key = keys[int(rng.integers(0, len(keys)))] if len(keys) > 1 else keys[0]
        bucket = self._records[key]
        donor = bucket[int(rng.integers(0, len(bucket)))]
        donor_length = donor.shape[0]
        if donor_length == length:
            return donor.copy()
        if length <= 2:
            if donor_length >= length:
                start = int(rng.integers(0, donor_length - length + 1))
                return donor[start : start + length].copy()
            indices = np.arange(length) % donor_length
            return donor[indices].copy()
        if donor_length >= length:
            start = int(rng.integers(0, donor_length - length + 1))
            return donor[start : start + length].copy()
        indices = np.arange(length) % donor_length
        return donor[indices].copy()


def _observation_mask(observed: np.ndarray) -> np.ndarray:
    values = np.asarray(observed)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("observed must have shape [T,6]")
    if values.dtype.kind == "b":
        return values.astype(bool, copy=True)
    if values.dtype.kind not in "iuf":
        raise ValueError("observed must be bool or a finite numeric binary mask")
    if not np.isfinite(values).all():
        raise ValueError("observed mask must not contain NaN or infinity")
    if not np.isin(values, (0, 1)).all():
        raise ValueError("observed mask must contain only 0 and 1")
    return values.astype(bool, copy=True)


def _validate_config(
    settings: Mapping[str, Any],
) -> tuple[dict[str, float], float, float, float, int, int]:
    weights = settings.get(
        "weights",
        {"random": 0.4, "historical": 0.4, "time_block": 0.1, "feature_block": 0.1},
    )
    if not isinstance(weights, Mapping):
        raise ValueError("weights must be a mapping")
    validated_weights: dict[str, float] = {}
    for name, raw_weight in weights.items():
        name = str(name)
        if name not in MASK_MODES:
            raise ValueError(f"unknown mask mode in weights: {name}")
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid weight for {name}") from exc
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"weight for {name} must be finite and nonnegative")
        validated_weights[name] = weight
    if not any(weight > 0 for weight in validated_weights.values()):
        raise ValueError("at least one mask weight must be positive")

    probability_range = np.asarray(
        settings.get("random_probability", [0.1, 0.5]), dtype=float
    ).reshape(-1)
    if probability_range.size != 2 or not np.isfinite(probability_range).all():
        raise ValueError("random_probability must be [low, high] finite values")
    p_low, p_high = float(probability_range[0]), float(probability_range[1])
    if not (0 <= p_low <= p_high <= 1):
        raise ValueError("random_probability must satisfy 0 <= low <= high <= 1")

    raw_min = settings.get("min_conditioned", 1)
    if isinstance(raw_min, bool) or not isinstance(raw_min, (int, np.integer)) or int(raw_min) < 0:
        raise ValueError("min_conditioned must be a nonnegative integer")
    min_conditioned = int(raw_min)
    try:
        block_fraction = float(settings.get("block_fraction", 0.3))
    except (TypeError, ValueError) as exc:
        raise ValueError("block_fraction must be in (0,1]") from exc
    if not math.isfinite(block_fraction) or not 0 < block_fraction <= 1:
        raise ValueError("block_fraction must be in (0,1]")
    raw_feature_count = settings.get("feature_count", settings.get("mask_feature_count", 1))
    if (
        isinstance(raw_feature_count, bool)
        or not isinstance(raw_feature_count, (int, np.integer))
        or int(raw_feature_count) < 1
    ):
        raise ValueError("feature_count must be a positive integer")
    return validated_weights, p_low, p_high, block_fraction, min_conditioned, int(raw_feature_count)


def _window_generator(
    rng: np.random.Generator, config: Mapping[str, Any], index: int
) -> np.random.Generator:
    """Give each row an independent stream while preserving prefix stability."""
    seeds = config.get("window_seeds")
    if seeds is not None:
        return np.random.default_rng(int(np.asarray(seeds).flat[index]))
    if "window_seed" in config:
        return np.random.default_rng(int(config["window_seed"]))
    return np.random.default_rng(int(rng.integers(0, 2**63 - 1)))


def _weighted_mode(local_rng: np.random.Generator, weights: Mapping[str, Any]) -> str:
    enabled = [(str(name), float(weight)) for name, weight in weights.items() if float(weight) > 0]
    if not enabled:
        return "random"
    names, values = zip(*enabled)
    probabilities = np.asarray(values, dtype=float)
    probabilities /= probabilities.sum()
    return str(local_rng.choice(names, p=probabilities))


def build_masks(
    observed: np.ndarray,
    scene: np.ndarray,
    bank: MaskBank,
    rng: np.random.Generator,
    config: Mapping[str, Any],
    mode: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build disjoint condition/target masks from a single observed window."""
    presence = _observation_mask(observed)
    scenes = np.asarray(scene)
    if scenes.ndim != 1 or len(scenes) != presence.shape[0]:
        raise ValueError("scene must have shape [T]")
    settings = dict(config.get("masking", {})) if "masking" in config else dict(config or {})
    weights, p_low, p_high, block_fraction, min_conditioned, feature_count = _validate_config(settings)
    requested = None if mode is None else str(mode)
    if requested is not None and requested not in MASK_MODES:
        raise ValueError(f"unknown mask mode: {requested}")
    ensure_target = bool(settings.get("ensure_target", True))

    # One call represents a temporal window; all choices use a local RNG stream.
    local_rng = _window_generator(rng, settings, 0)
    chosen = requested or _weighted_mode(local_rng, weights)
    target = np.zeros_like(presence, dtype=bool)
    historical_fallbacks = 0
    length = presence.shape[0]
    if length:
        if chosen == "random":
            probability = float(local_rng.uniform(p_low, p_high))
            target = presence & (local_rng.random(presence.shape) < probability)
        elif chosen == "historical":
            run_start = 0
            for row in range(1, length + 1):
                changed = row == length or _scene_key(scenes[row]) != _scene_key(scenes[run_start])
                if not changed:
                    continue
                donor = bank.sample(scenes[run_start], row - run_start, local_rng)
                if donor is None:
                    historical_fallbacks += 1
                    probability = float(local_rng.uniform(p_low, p_high))
                    target[run_start:row] = presence[run_start:row] & (
                        local_rng.random((row - run_start, 6)) < probability
                    )
                else:
                    target[run_start:row] = presence[run_start:row] & ~donor
                run_start = row
        elif chosen == "time_block":
            width = max(1, min(length, int(math.ceil(length * block_fraction))))
            start = int(local_rng.integers(0, length - width + 1))
            target[start : start + width] = presence[start : start + width]
        elif chosen == "feature_block":
            available_features = np.flatnonzero(presence.any(axis=0))
            take = min(feature_count, len(available_features))
            if take:
                selected = local_rng.choice(available_features, size=take, replace=False)
                target[:, selected] = presence[:, selected]
        elif chosen == "low_observation":
            for row, values in enumerate(presence):
                features = np.flatnonzero(values)
                keep = min(2, len(features))
                if keep:
                    conditioned = local_rng.choice(features, size=keep, replace=False)
                    target[row] = values
                    target[row, conditioned] = False

        available_total = int(presence.sum())
        maximum = max(0, available_total - min_conditioned)
        selected = np.argwhere(target)
        if len(selected) > maximum:
            keep = local_rng.choice(len(selected), size=maximum, replace=False)
            target[:] = False
            for row, feature in selected[keep]:
                target[row, feature] = True
        elif len(selected) == 0 and ensure_target and maximum > 0:
            available = np.argwhere(presence)
            row, feature = available[int(local_rng.integers(0, len(available)))]
            target[row, feature] = True

    condition = presence & ~target
    info = {
        "mode": chosen,
        "observed_count": int(presence.sum()),
        "condition_count": int(condition.sum()),
        "target_count": int(target.sum()),
        "empty_observations": int((presence.sum(axis=1) == 0).sum()),
        "historical_fallbacks": historical_fallbacks,
    }
    return condition.astype(np.float32), target.astype(np.float32), info


__all__ = ["MaskBank", "VITAL_COLUMNS", "MASK_MODES", "build_masks"]
