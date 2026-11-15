"""Read prepared STIRDiff tables and form padded admission windows.

Tables contain subject/admission identifiers, observation offsets in minutes,
scene, age, gender, and six vital columns. ``time`` is ordered as
``[vital_observationoffset, scene_observationoffset]``. Reading preserves the
numeric table values; it does not fit or apply normalization statistics.
Window tensors include raw values, observation masks, and a padding mask.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


VITALS = ["heartrate", "sbp", "dbp", "resprate", "temperature", "spo2"]
MASK_COLS = [f"{name}_observed" for name in VITALS]


def read_table(
    path,
    input_mode="raw",
    require_labels=True,
    max_admissions=None,
    *,
    reindex_duplicate_row_ids=False,
    allow_shared_hadm_ids=False,
):
    """Read and validate the required columns without fitting preprocessing.

    ``raw`` mode derives masks from finite values when absent and retains NaN
    for unobserved cells. ``preimputed`` mode requires explicit masks and
    finite values. IDs are kept as strings to avoid numeric coercion.
    """
    required = {
        "subject_id", "hadm_id", "hospital_observationoffset",
        "scene_observationoffset", "vital_observationoffset", "scene",
        "age", "gender", *VITALS,
    }
    columns = pd.read_csv(path, nrows=0).columns
    missing = required - set(columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    use = required | {"__row_id", "death", *MASK_COLS}
    frame = pd.read_csv(
        Path(path),
        usecols=lambda name: name in use,
        dtype={"subject_id": "string", "hadm_id": "string", "__row_id": "string"},
        low_memory=False,
        float_precision="round_trip",
    )
    if "subject_id" not in frame or "hadm_id" not in frame:
        raise ValueError("subject_id and hadm_id are required")
    for name in ("subject_id", "hadm_id"):
        frame[name] = frame[name].astype("string")
    if frame[["subject_id", "hadm_id"]].isna().any().any():
        raise ValueError("missing IDs")

    source_rows = int(len(frame))
    has_row_id = "__row_id" in frame
    if not has_row_id:
        frame["__row_id"] = np.arange(source_rows, dtype=np.int64)
    else:
        text = frame["__row_id"].astype("string")
        integer_literal = text.str.fullmatch(r"\+?\d+").fillna(False)
        digits = text.str.lstrip("+").str.lstrip("0").replace("", "0")
        within_bound = integer_literal & (
            (digits.str.len() < 19)
            | (
                (digits.str.len() == 19)
                & (digits <= str(np.iinfo(np.int64).max))
            )
        )
        if bool(integer_literal.all()):
            if not bool(within_bound.all()):
                raise ValueError(
                    "__row_id must contain finite nonnegative integers within int64 bounds "
                    "(technical row identity; repeated subject_id/hadm_id records are allowed)"
                )
            frame["__row_id"] = pd.to_numeric(text, downcast="integer").astype(np.int64)
        else:
            # Preserve exact integer identity for decimal or scientific spellings.
            parsed = []
            for value in text:
                try:
                    number = Decimal(str(value))
                    valid = (
                        number.is_finite()
                        and number >= 0
                        and number == number.to_integral_value()
                        and number <= Decimal(np.iinfo(np.int64).max)
                    )
                except (InvalidOperation, ValueError):
                    valid = False
                if not valid:
                    break
                parsed.append(int(number))
            if len(parsed) != source_rows:
                raise ValueError(
                    "__row_id must contain finite nonnegative integers within int64 bounds "
                    "(technical row identity; repeated subject_id/hadm_id records are allowed)"
                )
            frame["__row_id"] = np.asarray(parsed, dtype=np.int64)

    duplicate_count = int(frame["__row_id"].duplicated().sum())
    if duplicate_count:
        if not reindex_duplicate_row_ids:
            raise ValueError(
                "__row_id must be unique nonnegative integer technical row identity; "
                "repeated subject_id/hadm_id records are allowed, but duplicate __row_id "
                "values require reindex_duplicate_row_ids=True"
            )
        frame["__source_row_id"] = frame["__row_id"].to_numpy(np.int64, copy=True)
        frame["__row_id"] = np.arange(len(frame), dtype=np.int64)

    numeric_columns = [
        "hospital_observationoffset", "scene_observationoffset",
        "vital_observationoffset", "scene", "age", "gender", *VITALS,
    ]
    if "death" in frame:
        numeric_columns.append("death")
    for name in numeric_columns:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")

    if require_labels and "death" not in frame:
        raise ValueError("death labels required")
    if "death" in frame and (
        frame.death.isna().any() or not frame.death.isin([0, 1]).all()
    ):
        raise ValueError("death must be binary")
    if (
        frame.scene.isna().any()
        or (frame.scene % 1 != 0).any()
        or not frame.scene.isin([1, 2, 3, 4, 5]).all()
    ):
        raise ValueError("scene must be integer 1..5")
    if (
        frame.hospital_observationoffset.isna().any()
        or not np.isfinite(frame.hospital_observationoffset.to_numpy(float)).all()
    ):
        raise ValueError("hospital_observationoffset must be finite")

    if max_admissions is not None:
        if (
            isinstance(max_admissions, bool)
            or int(max_admissions) != max_admissions
            or max_admissions < 1
        ):
            raise ValueError("max_admissions must be positive integer")
        if allow_shared_hadm_ids:
            pairs = frame[["subject_id", "hadm_id"]].drop_duplicates().iloc[:int(max_admissions)]
            frame = frame.merge(pairs, on=["subject_id", "hadm_id"], how="inner", sort=False)
        else:
            ids = frame.hadm_id.drop_duplicates().iloc[:int(max_admissions)]
            frame = frame[frame.hadm_id.isin(ids)].copy()
    if frame.empty:
        raise ValueError("Empty input")

    shared_hadm = bool(frame.groupby("hadm_id", sort=False).subject_id.nunique().max() > 1)
    if shared_hadm and not allow_shared_hadm_ids:
        raise ValueError("hadm_id maps to multiple patients")

    mode = str(input_mode).lower()
    present = [name in frame for name in MASK_COLS]
    if any(present) and not all(present):
        raise ValueError("partial observed-mask columns")
    if mode not in {"raw", "preimputed"}:
        raise ValueError("input_mode must be raw or preimputed")
    if mode == "preimputed" and not all(present):
        raise ValueError("preimputed requires all observed masks")
    if all(present):
        for name in MASK_COLS:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
        if frame[MASK_COLS].isna().any().any() or not frame[MASK_COLS].isin([0, 1]).all().all():
            raise ValueError("masks must be strict 0/1")
        mask = frame[MASK_COLS].to_numpy(float)
        values = frame[VITALS].to_numpy(float)
        if mode == "preimputed" and not np.isfinite(values).all():
            raise ValueError("preimputed vitals must be finite")
        if np.any((mask == 1) & ~np.isfinite(values)):
            raise ValueError("observed vitals must be finite")
    else:
        mask = np.isfinite(frame[VITALS].to_numpy(float)).astype(np.float32)
        for index, name in enumerate(VITALS):
            frame[name + "_observed"] = mask[:, index]
    if mode == "raw":
        frame[VITALS] = frame[VITALS].where(frame[MASK_COLS].to_numpy(bool), np.nan)

    frame["__input_mode"] = mode
    frame = frame.reset_index(drop=True)
    frame.attrs["row_id_handling"] = {
        "policy": "reindexed_duplicate_ids" if duplicate_count else "validated",
        "rows": source_rows,
        "retained_rows": int(len(frame)),
        "duplicate_count": duplicate_count,
        "reindexed": bool(duplicate_count),
        "original_column": "__row_id" if duplicate_count else None,
        "preserved_column": "__source_row_id" if duplicate_count else None,
    }
    if shared_hadm:
        frame.attrs["identity_handling"] = {
            "shared_hadm_ids": True,
            "grouping": "subject_id+hadm_id",
        }
    return frame


def _ordered(frame):
    """Order rows within admissions while preserving source order for ties."""
    return frame.sort_values(
        ["hadm_id", "subject_id", "hospital_observationoffset", "__row_id"],
        kind="mergesort",
    ).reset_index(drop=True)


class WindowDataset(Dataset):
    """Split ordered admissions into windows and expose raw values plus masks."""

    def __init__(self, frame, max_length=128):
        if (
            frame.empty
            or isinstance(max_length, (bool, np.bool_))
            or not isinstance(max_length, (int, np.integer))
            or max_length < 1
        ):
            raise ValueError("empty frame or max_length must be a positive integer")
        ordered = _ordered(frame)
        self.max_length = int(max_length)
        self.arr = {
            "vitals": ordered[VITALS].to_numpy(float, copy=True),
            "observed": ordered[MASK_COLS].to_numpy(np.float32, copy=True),
            "age": ordered.age.to_numpy(float, copy=True),
            "gender": np.where(ordered.gender.isin([0, 1, 2]), ordered.gender, 2).astype(np.int64),
            "scene": ordered.scene.to_numpy(np.int64, copy=True),
            "time": ordered[["vital_observationoffset", "scene_observationoffset"]].to_numpy(float, copy=True),
            "hospital_offset": ordered.hospital_observationoffset.to_numpy(float, copy=True),
            "row_id": ordered["__row_id"].to_numpy(np.int64, copy=True),
        }
        # Zero is a tensor placeholder only; ``observed`` identifies measured values.
        self.arr["vitals"][~np.isfinite(self.arr["vitals"])] = 0.0
        self.labels = ordered.death.to_numpy(np.float32) if "death" in ordered else None

        self.starts = []
        self.ends = []
        admission_ids = ordered.hadm_id.to_numpy()
        subject_ids = ordered.subject_id.to_numpy()
        bounds = np.r_[
            0,
            np.flatnonzero(
                (admission_ids[1:] != admission_ids[:-1])
                | (subject_ids[1:] != subject_ids[:-1])
            ) + 1,
            len(ordered),
        ]
        for start, end in zip(bounds[:-1], bounds[1:]):
            self.starts.extend(range(int(start), int(end), self.max_length))
            self.ends.extend(
                min(index + self.max_length, int(end))
                for index in range(int(start), int(end), self.max_length)
            )

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start, end = int(self.starts[index]), int(self.ends[index])
        item = {
            name: torch.from_numpy(np.asarray(values[start:end]).copy())
            for name, values in self.arr.items()
        }
        item["window_id"] = torch.full((end - start,), int(index), dtype=torch.long)
        if self.labels is not None:
            item["labels"] = torch.from_numpy(self.labels[start:end].copy())
        return item


def collate_windows(items):
    """Pad variable-length windows and return lengths plus a valid-row mask."""
    lengths = torch.tensor([len(item["vitals"]) for item in items], dtype=torch.long)
    output = {
        "lengths": lengths,
        "valid": torch.arange(int(lengths.max()))[None, :] < lengths[:, None],
    }
    for name, value in items[0].items():
        if isinstance(value, torch.Tensor):
            output[name] = pad_sequence(
                [item[name] for item in items], batch_first=True, padding_value=0
            )
    return output


__all__ = ["VITALS", "MASK_COLS", "read_table", "WindowDataset", "collate_windows"]
