"""Imputation and prediction on input tables, with row alignment preserved."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .data import VITALS, WindowDataset, collate_windows, read_table
from .model import load_checkpoint

def complete_frame(model, frame, batch_size=64):
    """Generate using full-cohort generator windows, before predictor rewindowing."""
    index = pd.Index(frame["__row_id"])
    if frame.empty or index.has_duplicates:
        raise ValueError("Completion requires nonempty input with unique row IDs")
    ds = WindowDataset(frame, int(model.inference["max_length"]))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_windows)
    device = next(model.parameters()).device
    values = np.empty((len(frame), len(VITALS)), dtype=np.float64)
    seen = np.zeros(len(frame), dtype=bool)
    model.eval()
    with torch.no_grad():
        for raw in loader:
            b = {k:v.to(device) for k,v in raw.items()}
            valid = b["valid"]
            positions = index.get_indexer(b["row_id"][valid].cpu().numpy())
            if (positions < 0).any() or seen[positions].any():
                raise ValueError("Invalid completion row alignment")
            values[positions] = model.complete(b, gradient=False)[valid].cpu().numpy()
            seen[positions] = True
    if not seen.all() or not np.isfinite(values).all():
        raise ValueError("Incomplete or nonfinite imputation output")
    result = frame.copy()
    result[VITALS] = values
    return result


def predict_frame(model, frame, input_mode="raw", batch_size=64):
    """Predict sorted admission windows and restore original row order."""
    if frame.empty:
        raise ValueError("Empty inference frame")
    if input_mode not in {"raw", "preimputed"}:
        raise ValueError("Unknown input mode")
    if input_mode == "raw" and model.prediction_max_length != model.inference["max_length"]:
        frame = complete_frame(model, frame, batch_size)
        input_mode = "preimputed"
    device = next(model.parameters()).device
    ds = WindowDataset(frame, int(model.prediction_max_length))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_windows)
    ids, probabilities = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            prob = model(batch, input_mode=input_mode).sigmoid()
            valid = batch["valid"]
            ids.append(batch["row_id"][valid].cpu().numpy())
            probabilities.append(prob[valid].cpu().numpy())
    ids, probabilities = np.concatenate(ids), np.concatenate(probabilities)
    if len(ids) != len(frame) or len(np.unique(ids)) != len(ids):
        raise ValueError("Missing or duplicate prediction row IDs")
    lookup = pd.Series(probabilities, index=ids)
    restored = lookup.reindex(frame["__row_id"].to_numpy()).to_numpy()
    if not np.isfinite(restored).all() or ((restored < 0) | (restored > 1)).any():
        raise ValueError("Invalid or missing probability")
    cols = [c for c in ("__row_id", "subject_id", "hadm_id", "scene",
                       "hospital_observationoffset", "scene_observationoffset",
                       "vital_observationoffset", "death") if c in frame]
    out = frame[cols].copy().rename(columns={"death": "true_label"})
    out["risk_probability"] = restored
    return out



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists; choose a new filename")
    model, _ = load_checkpoint(args.checkpoint, args.device)
    frame = read_table(args.input, require_labels=False)
    predictions = predict_frame(model, frame, batch_size=args.batch_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)


if __name__ == "__main__":
    main()