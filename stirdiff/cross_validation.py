"""Patient-grouped, fixed five-fold cross-validation for STIRDiff."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .data import read_table
from .imputer import ImputerConfig
from .predictor import PredictorConfig
from .train import TrainingConfig, add_training_arguments, config_from_args, train


def _jsonable(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _canonical(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else str(value)
    return str(value)


def _admissions(frame):
    required = {"subject_id", "hadm_id", "death"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("development frame must be nonempty")
    for col in ("subject_id", "hadm_id"):
        values = frame[col].astype("string")
        if values.isna().any() or values.str.strip().eq("").any():
            raise ValueError(f"{col} values must be present and nonempty")
    labels = pd.to_numeric(frame.death, errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("death labels must be binary")
    f = frame.copy()
    f["subject_id"] = f.subject_id.astype(str)
    f["hadm_id"] = f.hadm_id.astype(str)
    f["death"] = labels.astype(np.int8)
    # hadm_id is required to identify one admission globally; a split pair identity
    # alone would allow one encounter to appear under two patient IDs.
    if f.groupby("hadm_id", sort=False).subject_id.nunique().gt(1).any():
        raise ValueError("hadm_id must map to exactly one subject_id")
    grouped = f.groupby(["subject_id", "hadm_id"], sort=False).death
    inconsistent = grouped.nunique().gt(1)
    if inconsistent.any():
        raise ValueError("death outcome must be constant within each admission")
    admissions = (f.groupby(["subject_id", "hadm_id"], sort=False).death.first()
                  .rename("outcome").reset_index())
    admissions = admissions.sort_values(["subject_id", "hadm_id"], kind="mergesort").reset_index(drop=True)
    counts = admissions.outcome.value_counts()
    if len(counts) != 2 or any(int(counts.get(c, 0)) < 5 for c in (0, 1)):
        raise ValueError("five-fold stratification requires at least five admissions of each outcome")
    if admissions.subject_id.nunique() < 5:
        raise ValueError("five-fold split requires at least five distinct subjects")
    return f, admissions


def _source_hash(frame):
    if "__row_id" not in frame:
        raise ValueError("frame must include validated unique __row_id values")
    ids = pd.to_numeric(frame["__row_id"], errors="coerce")
    if (not np.isfinite(ids).all() or ids.duplicated().any() or (ids < 0).any()
            or (ids % 1 != 0).any() or (ids > np.iinfo(np.int64).max).any()):
        raise ValueError("__row_id must contain unique nonnegative int64 values")
    cols = sorted(str(c) for c in frame.columns if str(c) != "__input_mode")
    ordered = frame.assign(__cv_hash_row_id=ids).sort_values("__cv_hash_row_id", kind="mergesort")
    digest = hashlib.sha256()
    header = json.dumps(cols, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    for row in ordered[cols].itertuples(index=False, name=None):
        encoded = json.dumps([_canonical(v) for v in row], ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fold_is_complete(directory, manifest_digest, fold_number):
    marker = directory / "cv_complete.json"
    if not directory.is_dir() or not marker.is_file():
        return False
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
        if record.get("manifest_sha256") != manifest_digest or record.get("fold_number") != fold_number:
            return False
        files = record.get("files")
        if not isinstance(files, dict) or not files:
            return False
        required = {
            "model.pt", "history.json",
            "stages/generator_warmup/model.pt",
            "stages/frozen_predictor/model.pt",
            "stages/joint/model.pt",
        }
        if not required.issubset(files):
            return False
        for name, digest in files.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or not isinstance(digest, str):
                return False
            candidate = directory / relative
            if not candidate.is_file() or _file_hash(candidate) != digest:
                return False
        return True
    except (OSError, ValueError, TypeError):
        return False


def _requested_folds(folds):
    if folds is None:
        return [1, 2, 3, 4, 5]
    if isinstance(folds, (int, np.integer)) and not isinstance(folds, (bool, np.bool_)):
        folds = [int(folds)]
    result = list(folds)
    if not result or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or not 1 <= int(x) <= 5 for x in result):
        raise ValueError("folds must be a nonempty sequence of fold numbers in 1..5")
    result = [int(x) for x in result]
    if len(set(result)) != len(result):
        raise ValueError("folds must not contain duplicates")
    return result


def cross_validate(frame, output, config=None, folds=None, split_seed=2026,
                   generator_config=None, predictor_config=None, device="cpu"):
    """Train selected folds from one supplied development table, grouped by patient."""
    config = (config or TrainingConfig()).validate()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if isinstance(split_seed, bool) or int(split_seed) != split_seed or split_seed < 0:
        raise ValueError("split_seed must be a nonnegative integer")
    requested = _requested_folds(folds)
    data, admissions = _admissions(frame)
    generator_config = generator_config or ImputerConfig()
    predictor_config = predictor_config or PredictorConfig()

    # Generate and validate all five splits before creating fold directories or training.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=int(split_seed))
    partitions = {}
    for number, (train_idx, dev_idx) in enumerate(
        splitter.split(admissions, admissions.outcome, groups=admissions.subject_id), start=1
    ):
        train_adm = admissions.iloc[train_idx]
        dev_adm = admissions.iloc[dev_idx]
        train_subjects = set(train_adm.subject_id)
        dev_subjects = set(dev_adm.subject_id)
        if not train_subjects or not dev_subjects or train_subjects & dev_subjects:
            raise ValueError(f"fold {number} has empty partitions or patient overlap")
        train_labels = set(train_adm.outcome.astype(int))
        dev_labels = set(dev_adm.outcome.astype(int))
        if train_labels != {0, 1}:
            raise ValueError(f"fold {number} training partition must contain both outcome classes")
        dev_frame_adm = data.merge(dev_adm[["subject_id", "hadm_id"]], on=["subject_id", "hadm_id"], how="inner", sort=False)
        for scene in config.selection_scenes:
            rows = dev_frame_adm[pd.to_numeric(dev_frame_adm.scene, errors="coerce") == scene]
            if rows.empty or rows.death.nunique() != 2:
                raise ValueError(f"fold {number} heldout data must contain both classes in selection scene {scene}")
        partitions[number] = {
            "train_admissions": train_adm[["subject_id", "hadm_id", "outcome"]].to_dict("records"),
            "dev_admissions": dev_adm[["subject_id", "hadm_id", "outcome"]].to_dict("records"),
            "train_subjects": sorted(train_subjects), "dev_subjects": sorted(dev_subjects),
            "train_count": int(len(train_adm)), "dev_count": int(len(dev_adm)),
        }

    manifest = {
        "format_version": 1, "fold_count": 5,
        "source_sha256": _source_hash(frame), "split_seed": int(split_seed),
        "training_config": _jsonable(config),
        "generator_config": _jsonable(generator_config),
        "predictor_config": _jsonable(predictor_config),
        "partitions": partitions,
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "split_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_digest = hashlib.sha256(json.dumps(existing, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        if existing_digest != manifest_digest:
            raise ValueError("output root contains a split manifest for different data or configuration")
    else:
        if any(output.glob("fold_*")):
            raise ValueError("fold directories exist without a split manifest")
        _atomic_json(manifest_path, manifest)

    status_path = output / "cv_status.json"
    completed = []
    summaries = {}
    for number in range(1, 6):
        folder = output / f"fold_{number}"
        if _fold_is_complete(folder, manifest_digest, number):
            completed.append(number)
            try:
                hist = json.loads((folder / "history.json").read_text(encoding="utf-8"))
                summaries[str(number)] = {"threshold": hist.get("threshold"), "selection_score": hist.get("selection_score")}
            except (OSError, ValueError):
                pass

    # Validate every requested existing directory before starting any new fold.
    for number in requested:
        folder = output / f"fold_{number}"
        if folder.exists() and number not in completed:
            raise FileExistsError(f"fold directory exists but is incomplete or mismatched: {folder}")

    # Persist an accurate partial state before any newly requested training starts.
    _atomic_json(status_path, {
        "status": "complete" if len(completed) == 5 else "partial",
        "requested_folds": requested, "completed_folds": sorted(set(completed)),
        "summaries": summaries,
    })

    for number in requested:
        folder = output / f"fold_{number}"
        if number in completed:
            continue
        _atomic_json(status_path, {
            "status": "partial", "requested_folds": requested,
            "completed_folds": sorted(set(completed)), "summaries": summaries,
        })
        part = partitions[number]
        keys_train = pd.DataFrame(part["train_admissions"])[["subject_id", "hadm_id"]]
        keys_dev = pd.DataFrame(part["dev_admissions"])[["subject_id", "hadm_id"]]
        train_frame = data.merge(keys_train, on=["subject_id", "hadm_id"], how="inner", sort=False)
        dev_frame = data.merge(keys_dev, on=["subject_id", "hadm_id"], how="inner", sort=False)
        # Deterministic canonical row order ensures shuffled source rows train identically.
        order = [c for c in ("subject_id", "hadm_id", "hospital_observationoffset", "__row_id") if c in data.columns]
        if order:
            train_frame = train_frame.sort_values(order, kind="mergesort").reset_index(drop=True)
            dev_frame = dev_frame.sort_values(order, kind="mergesort").reset_index(drop=True)
        folder.mkdir(parents=True)
        try:
            _, history = train(train_frame, dev_frame, folder, config, generator_config,
                               predictor_config, device=device)
            expected = [p for p in folder.rglob("*") if p.is_file() and p.name != "cv_complete.json"]
            files = {p.relative_to(folder).as_posix(): _file_hash(p) for p in sorted(expected)}
            required = {"model.pt", "history.json", "stages/generator_warmup/model.pt",
                        "stages/frozen_predictor/model.pt", "stages/joint/model.pt"}
            if not required.issubset(files):
                raise RuntimeError("training returned without expected model and history files")
            _atomic_json(folder / "cv_complete.json", {
                "manifest_sha256": manifest_digest, "fold_number": number, "files": files,
            })
            completed.append(number)
            summaries[str(number)] = {"threshold": history.get("threshold"), "selection_score": history.get("selection_score")}
        except Exception:
            # Existing fold directory intentionally remains as a partial-run marker.
            _atomic_json(status_path, {
                "status": "failed", "requested_folds": requested,
                "completed_folds": sorted(set(completed)), "failed_fold": number,
                "summaries": summaries,
            })
            raise
        _atomic_json(status_path, {
            "status": "complete" if len(completed) == 5 else "partial",
            "requested_folds": requested, "completed_folds": sorted(set(completed)),
            "summaries": summaries,
        })
    result = {
        "status": "complete" if len(completed) == 5 else "partial",
        "requested_folds": requested, "completed_folds": sorted(set(completed)),
        "summaries": summaries,
    }
    _atomic_json(status_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run patient-grouped fixed five-fold STIRDiff cross-validation")
    parser.add_argument("--data", required=True, type=Path, help="one prepared development CSV; no external test table")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--folds", type=int, nargs="+", choices=range(1, 6), default=[1, 2, 3, 4, 5])
    parser.add_argument("--split-seed", type=int, default=2026)
    add_training_arguments(parser)
    args = parser.parse_args(argv)
    if len(set(args.folds)) != len(args.folds):
        parser.error("--folds values must be unique")
    config = config_from_args(args)
    result = cross_validate(read_table(args.data, input_mode="raw", require_labels=True),
                            args.output, config=config, folds=args.folds,
                            split_seed=args.split_seed, device=args.device)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = ["cross_validate", "main"]
