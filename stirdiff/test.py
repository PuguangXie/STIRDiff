"""Run predictions and calculate basic metrics on a labelled test table."""
import argparse
import json
from pathlib import Path

import torch

from .data import read_table
from .metrics import basic_metrics
from .model import load_checkpoint
from .predict import predict_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch-size and threads must be positive")
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists; choose a new directory")
    torch.set_num_threads(args.threads)
    model, threshold = load_checkpoint(args.checkpoint, args.device)
    if threshold is None:
        parser.error("checkpoint has no development-set Youden cutoff; use a risk-trained checkpoint")
    frame = read_table(args.input)
    predictions = predict_frame(model, frame, batch_size=args.batch_size)
    def summarize(rows):
        return basic_metrics(rows.true_label.to_numpy(), rows.risk_probability.to_numpy(), threshold)
    report = {
        "overall": summarize(predictions),
        "by_scene": {str(int(scene)): summarize(rows)
                     for scene, rows in predictions.groupby("scene", sort=True)},
        "threshold": threshold,
        "threshold_source": "development_youden",
        "threshold_rule": "probability >= threshold",
    }
    output.mkdir(parents=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
