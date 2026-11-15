"""Three-stage STIRDiff training: imputer warmup, frozen predictor, then joint."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, is_dataclass
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import MASK_COLS, VITALS, WindowDataset, collate_windows, read_table
from .imputer import ImputerConfig
from .masking import MaskBank, build_masks
from .metrics import basic_metrics, youden_threshold
from .model import IntegratedModel, save_checkpoint
from .normalization import fit_statistics
from .predict import complete_frame
from .predictor import PredictorConfig


@dataclass
class TrainingConfig:
    """Conventional defaults for a new run; all stage limits are configurable."""

    warmup_epochs: int = 3
    predictor_epochs: int = 6
    joint_epochs: int = 3
    batch_size: int = 64
    eval_batch_size: int = 64
    max_length: int = 512
    steps: int = 6
    samples: int = 5
    seed: int = 2026
    patience: int = 2
    min_delta: float = 2e-4
    selection_scenes: tuple[int, ...] = (1, 2, 3, 5)
    warmup_learning_rate: float = 3e-5
    predictor_learning_rate: float = 1e-4
    joint_generator_learning_rate: float = 1e-5
    joint_predictor_learning_rate: float = 3e-5
    risk_loss_weight: float = 1.0
    diffusion_loss_weight: float = 0.02
    weight_decay: float = 1e-5
    gradient_clip: float = 5.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 1
    scheduler_min_lr: float = 1e-6
    num_workers: int = 0
    threads: int = 2
    mask_bank_per_scene: int = 1024
    masking: dict = field(default_factory=lambda: {
        "weights": {"random": 0.4, "historical": 0.4, "time_block": 0.1, "feature_block": 0.1},
        "random_probability": [0.1, 0.5],
        "block_fraction": 0.3,
        "min_conditioned": 1,
        "ensure_target": True,
        "feature_count": 1,
    })

    def validate(self):
        for name in ("warmup_epochs", "predictor_epochs", "joint_epochs", "batch_size",
                     "eval_batch_size", "max_length", "steps", "samples", "patience", "threads",
                     "mask_bank_per_scene"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.steps < 2:
            raise ValueError("steps must be at least 2")
        if isinstance(self.seed, bool) or int(self.seed) != self.seed or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if isinstance(self.num_workers, bool) or int(self.num_workers) != self.num_workers or self.num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer")
        if self.risk_loss_weight <= 0 or self.diffusion_loss_weight <= 0:
            raise ValueError("joint training requires positive risk and diffusion loss weights")
        if (isinstance(self.scheduler_patience, bool)
                or int(self.scheduler_patience) != self.scheduler_patience
                or self.scheduler_patience < 0):
            raise ValueError("scheduler_patience must be a nonnegative integer")
        if len(set(self.selection_scenes)) != len(self.selection_scenes) or not self.selection_scenes:
            raise ValueError("selection_scenes must be a nonempty list of unique scene IDs")
        if any(isinstance(scene, bool) or int(scene) != scene or not 1 <= scene <= 5
               for scene in self.selection_scenes):
            raise ValueError("selection_scenes must contain scene IDs in 1..5")
        for name in ("min_delta", "risk_loss_weight", "diffusion_loss_weight", "weight_decay"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("warmup_learning_rate", "predictor_learning_rate", "joint_generator_learning_rate",
                     "joint_predictor_learning_rate", "gradient_clip", "scheduler_factor", "scheduler_min_lr"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.scheduler_factor < 1:
            raise ValueError("scheduler_factor must be in (0, 1)")
        return self


def _plain_config(value):
    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    return dict(value)


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(device=None):
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _loader(dataset, config, seed, shuffle):
    lengths = np.asarray(dataset.ends) - np.asarray(dataset.starts)
    edges = np.array([1 << i for i in range((int(dataset.max_length) - 1).bit_length() + 1)])
    buckets = np.searchsorted(edges, lengths)
    rng = np.random.default_rng(seed)
    batches = []
    for bucket in np.unique(buckets):
        ids = np.flatnonzero(buckets == bucket)
        if shuffle:
            ids = ids[rng.permutation(len(ids))]
        batches.extend(ids[i:i + config.batch_size].tolist()
                       for i in range(0, len(ids), config.batch_size))
    if shuffle and batches:
        batches = [batches[i] for i in rng.permutation(len(batches))]
    return DataLoader(
        dataset, batch_sampler=batches, collate_fn=collate_windows,
        num_workers=config.num_workers,
        generator=torch.Generator().manual_seed(int(seed)),
    )


def _move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def _observed_frame(frame):
    result = frame.copy()
    if all(name in result for name in MASK_COLS):
        mask = result[MASK_COLS].to_numpy(bool)
    else:
        mask = np.isfinite(result[VITALS].to_numpy(float))
    result[VITALS] = result[VITALS].where(mask, np.nan)
    return result


def _scene_weights(frame, device):
    counts = frame.groupby("scene").size()
    weights = torch.zeros(6, dtype=torch.float32, device=device)
    for scene, count in counts.items():
        weights[int(scene)] = len(frame) / (len(counts) * int(count))
    return weights


def _make_masks(batch, bank, config, epoch, *, validation=False):
    conditions = torch.zeros_like(batch["observed"], dtype=torch.float32)
    targets = torch.zeros_like(conditions)
    base_seed = int(config.seed) + (200000 if validation else 0)
    mask_epoch = 0 if validation else int(epoch)
    for index, length in enumerate(batch["lengths"].detach().cpu().tolist()):
        window_id = int(batch["window_id"][index, 0])
        rng = np.random.default_rng(np.random.SeedSequence([base_seed, mask_epoch, window_id]))
        condition, target, _ = build_masks(
            batch["observed"][index, :length].detach().cpu().numpy(),
            batch["scene"][index, :length].detach().cpu().numpy(),
            bank, rng, config.masking,
        )
        conditions[index, :length] = torch.as_tensor(condition, device=conditions.device)
        targets[index, :length] = torch.as_tensor(target, device=targets.device)
    return conditions, targets


def _selection_score(labels, probabilities, scenes, selection_scenes):
    scores = []
    for scene in selection_scenes:
        selected = scenes == scene
        if not selected.any() or len(np.unique(labels[selected])) != 2:
            raise ValueError(
                f"Selection scene {scene} must contain both label classes in development data; "
                "choose selection_scenes explicitly for this dataset"
            )
        scores.append(float(basic_metrics(labels[selected], probabilities[selected])["auroc"]))
    return float(np.mean(scores))


def _risk_validation(model, loader, config, device, input_mode):
    model.eval()
    ys, ps, ss = [], [], []
    with torch.no_grad():
        for raw in loader:
            batch = _move(raw, device)
            probabilities = model(batch, input_mode=input_mode).sigmoid()
            valid = batch["valid"]
            ys.append(batch["labels"][valid].cpu().numpy())
            ps.append(probabilities[valid].cpu().numpy())
            ss.append(batch["scene"][valid].cpu().numpy())
    labels, probabilities, scenes = map(np.concatenate, (ys, ps, ss))
    score = _selection_score(labels, probabilities, scenes, config.selection_scenes)
    threshold = youden_threshold(labels, probabilities)
    return score, threshold, {"dev_macro_auroc": score,
                              "development_youden_threshold": threshold}


def _generator_validation(model, loader, config, bank, device):
    model.eval()
    losses = []
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
        torch.manual_seed(int(config.seed) + 100000)
        for raw in loader:
            batch = _move(raw, device)
            condition, target = _make_masks(batch, bank, config, 0, validation=True)
            loss, info = model.diffusion_loss(batch, condition, target)
            if info.get("target_count", 0):
                losses.append(float(loss.detach()))
    if not losses:
        raise ValueError("Development data produced no artificial-mask targets for generator selection")
    value = float(np.mean(losses))
    return -value, None, {"dev_masked_loss": value}


def _new_model(generator_config, predictor_config, generator_preprocessor,
               predictor_preprocessor, inference, freeze_generator):
    return IntegratedModel(
        generator_config, predictor_config, generator_preprocessor,
        predictor_preprocessor, inference, freeze_generator=freeze_generator,
    )


def _fit_stage(model, train_frame, dev_frame, output_dir, config, *, stage,
               epochs, device, input_mode, optimizer_mode, mask_bank=None,
               save_stage=True):
    """Fit one stage, restoring its best development-selected state in place."""
    train_dataset = WindowDataset(train_frame, config.max_length)
    dev_dataset = WindowDataset(dev_frame, config.max_length)
    train_loader_for_norm = _loader(train_dataset, config, config.seed, shuffle=False)
    dev_loader = _loader(dev_dataset, config, config.seed + 100000, shuffle=False)
    fixed_normalizer = len(train_frame) / max(1, len(train_loader_for_norm))
    weights = _scene_weights(train_frame, device)

    if optimizer_mode == "warmup":
        params = [{"params": list(model.generator.parameters()), "lr": config.warmup_learning_rate}]
    elif optimizer_mode == "predictor":
        model.freeze_generator = True
        model.generator.requires_grad_(False)
        model.predictor.requires_grad_(True)
        params = [{"params": list(model.predictor.parameters()), "lr": config.predictor_learning_rate}]
    else:
        model.freeze_generator = False
        model.generator.requires_grad_(True)
        model.predictor.requires_grad_(True)
        params = [
            {"params": list(model.generator.parameters()), "lr": config.joint_generator_learning_rate},
            {"params": list(model.predictor.parameters()), "lr": config.joint_predictor_learning_rate},
        ]
    optimizer = torch.optim.AdamW(params, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config.scheduler_factor,
        patience=config.scheduler_patience, min_lr=config.scheduler_min_lr,
    )
    best_score = -float("inf")
    best_threshold = None
    best_state = None
    stale = 0
    history = []
    epoch_limit = int(epochs)
    for epoch in range(1, epoch_limit + 1):
        loader = _loader(train_dataset, config, config.seed + epoch, shuffle=True)
        normalizer = fixed_normalizer
        model.train()
        sums = {"total": 0.0, "risk": 0.0, "diffusion": 0.0}
        batches = 0
        for raw in loader:
            batch = _move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            zero = torch.zeros((), device=device)
            risk_loss = zero
            diffusion_loss = zero
            if optimizer_mode != "warmup":
                logits = model(batch, input_mode=input_mode)
                valid = batch["valid"]
                cell_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[valid], batch["labels"][valid].float(), reduction="none"
                )
                risk_loss = (cell_loss * weights[batch["scene"][valid].long()]).sum() / normalizer
            if optimizer_mode != "predictor":
                condition, target = _make_masks(batch, mask_bank, config, epoch)
                diffusion_loss, _ = model.diffusion_loss(batch, condition, target)
            total = (diffusion_loss if optimizer_mode == "warmup" else
                     risk_loss if optimizer_mode == "predictor" else
                     config.risk_loss_weight * risk_loss + config.diffusion_loss_weight * diffusion_loss)
            if not torch.isfinite(total):
                raise FloatingPointError(f"Nonfinite objective in {stage}")
            total.backward()
            parameters = [p for group in params for p in group["params"] if p.grad is not None]
            torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            batches += 1
            sums["total"] += float(total.detach())
            sums["risk"] += float(risk_loss.detach())
            sums["diffusion"] += float(diffusion_loss.detach())

        if optimizer_mode == "warmup":
            score, threshold, dev_record = _generator_validation(model, dev_loader, config, mask_bank, device)
        else:
            score, threshold, dev_record = _risk_validation(
                model, dev_loader, config, device, input_mode
            )
        scheduler.step(score)
        record = {
            "stage": stage, "epoch": epoch, "train_loss": sums["total"] / max(1, batches),
            "risk_loss": sums["risk"] / max(1, batches),
            "diffusion_loss": sums["diffusion"] / max(1, batches), **dev_record,
        }
        history.append(record)
        if score > best_score + config.min_delta:
            best_score = score
            best_threshold = threshold
            stale = 0
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= config.patience:
            break

    if best_state is None:
        raise RuntimeError(f"No best checkpoint selected for stage {stage}")
    model.load_state_dict(best_state, strict=True)
    model.to(device)
    if save_stage:
        stage_dir = Path(output_dir) / "stages" / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(stage_dir / "model.pt", model, threshold=best_threshold)
    return model, best_threshold, best_score, history


def _model_configs(generator_config=None, predictor_config=None):
    generator = _plain_config(generator_config) or asdict(ImputerConfig())
    predictor = _plain_config(predictor_config) or asdict(PredictorConfig())
    return generator, predictor


def _inference(config):
    return {
        "steps": int(config.steps), "samples": int(config.samples),
        "max_length": int(config.max_length), "predictor_max_length": int(config.max_length),
        "seed": int(config.seed), "aggregation": "mean",
    }


def _assert_disjoint_subjects(train_frame, dev_frame):
    if "subject_id" not in train_frame or "subject_id" not in dev_frame:
        raise ValueError("train and dev frames must contain subject_id")
    overlap = set(train_frame.subject_id.astype(str)) & set(dev_frame.subject_id.astype(str))
    if overlap:
        raise ValueError("train and development subject_id values overlap")


def _validate_labels_and_scenes(train_frame, dev_frame, selection_scenes):
    for name, frame in (("train", train_frame), ("development", dev_frame)):
        labels = pd.to_numeric(frame["death"], errors="coerce").to_numpy()
        if not np.isfinite(labels).all() or not np.isin(labels, (0, 1)).all():
            raise ValueError(f"{name} death labels must be finite binary values")
    scenes = pd.to_numeric(dev_frame["scene"], errors="coerce").to_numpy()
    if not np.isfinite(scenes).all():
        raise ValueError("development scene values must be finite")
    labels = dev_frame["death"].to_numpy(dtype=np.int8)
    for scene in selection_scenes:
        selected = scenes == scene
        if not selected.any() or len(np.unique(labels[selected])) != 2:
            raise ValueError(
                f"Selection scene {scene} must contain both label classes in development data; "
                "choose selection_scenes explicitly for this dataset"
            )


def train(train_frame, dev_frame, output, config=None, generator_config=None,
          predictor_config=None, *, device=None):
    """Run all three training stages from raw train/dev frames and save the best model.

    Development data selects epochs and the final Youden cutoff. It is never
    used to fit normalization statistics or the artificial-mask bank.
    """
    config = (config or TrainingConfig()).validate()
    if not isinstance(train_frame, pd.DataFrame) or not isinstance(dev_frame, pd.DataFrame):
        raise TypeError("train_frame and dev_frame must be pandas DataFrames")
    if train_frame.empty or dev_frame.empty:
        raise ValueError("train_frame and dev_frame must be nonempty")
    if "death" not in train_frame or "death" not in dev_frame:
        raise ValueError("train and dev frames must include binary death labels")
    _assert_disjoint_subjects(train_frame, dev_frame)
    _validate_labels_and_scenes(train_frame, dev_frame, config.selection_scenes)
    train_frame = train_frame.reset_index(drop=True).copy()
    dev_frame = dev_frame.reset_index(drop=True).copy()
    train_raw = _observed_frame(train_frame)
    dev_raw = _observed_frame(dev_frame)
    generator_config, predictor_config = _model_configs(generator_config, predictor_config)
    inference = _inference(config)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.threads)
    run_device = _device(device)
    _seed(config.seed)

    # Stage 1: fit normalization on observed training values, then train only G.
    generator_statistics = fit_statistics(train_raw)
    placeholder_predictor_statistics = generator_statistics
    model = _new_model(generator_config, predictor_config, generator_statistics,
                       placeholder_predictor_statistics, inference, freeze_generator=False).to(run_device)
    mask_bank = MaskBank.fit(
        train_raw, seed=config.seed, per_scene=config.mask_bank_per_scene,
        max_length=config.max_length,
    )
    model, _, _, warmup_history = _fit_stage(
        model, train_raw, dev_raw, output, config, stage="generator_warmup",
        epochs=config.warmup_epochs, device=run_device, input_mode="raw",
        optimizer_mode="warmup", mask_bank=mask_bank,
    )

    # Materialize fills once per split. Only completed TRAIN rows fit predictor scales.
    filled_train = complete_frame(model, train_raw, batch_size=config.eval_batch_size)
    filled_dev = complete_frame(model, dev_raw, batch_size=config.eval_batch_size)
    predictor_statistics = fit_statistics(filled_train)
    generator_state = {key: value.detach().cpu().clone()
                       for key, value in model.generator.state_dict().items()}
    predictor_stage_model = _new_model(
        generator_config, predictor_config, generator_statistics,
        predictor_statistics, inference, freeze_generator=True,
    ).to(run_device)
    predictor_stage_model.generator.load_state_dict(generator_state, strict=True)
    model, predictor_threshold, _, predictor_history = _fit_stage(
        predictor_stage_model, filled_train, filled_dev, output, config,
        stage="frozen_predictor", epochs=config.predictor_epochs,
        device=run_device, input_mode="preimputed", optimizer_mode="predictor",
    )

    # Stage 3: resume both selected components and optimize the raw pipeline jointly.
    joint_model = _new_model(
        generator_config, predictor_config, generator_statistics,
        predictor_statistics, inference, freeze_generator=False,
    ).to(run_device)
    joint_model.generator.load_state_dict(model.generator.state_dict(), strict=True)
    joint_model.predictor.load_state_dict(model.predictor.state_dict(), strict=True)
    joint_model, threshold, selection_score, joint_history = _fit_stage(
        joint_model, train_raw, dev_raw, output, config, stage="joint",
        epochs=config.joint_epochs, device=run_device, input_mode="raw",
        optimizer_mode="joint", mask_bank=mask_bank,
    )
    save_checkpoint(output / "model.pt", joint_model, threshold=threshold)
    history = {
        "generator_warmup": warmup_history,
        "frozen_predictor": predictor_history,
        "joint": joint_history,
        "selected_stage": "joint",
        "selection_scenes": list(config.selection_scenes),
        "selection_metric": "mean scene AUROC",
        "threshold": threshold,
        "threshold_source": "development_youden",
        "threshold_rule": "probability >= threshold",
        "predictor_stage_threshold": predictor_threshold,
        "selection_score": selection_score,
        "training_config": asdict(config),
        "generator_config": generator_config,
        "predictor_config": predictor_config,
    }
    (output / "history.json").write_text(
        json.dumps(history, indent=2, allow_nan=False), encoding="utf-8"
    )
    return joint_model, history


def _read_cli(path):
    return read_table(path, input_mode="raw", require_labels=True)


def add_training_arguments(parser):
    """Shared options for a predefined split and five-fold training."""
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--predictor-epochs", type=int, default=6)
    parser.add_argument("--joint-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--selection-scenes", type=int, nargs="+", default=[1, 2, 3, 5])
    return parser


def config_from_args(args):
    return TrainingConfig(
        warmup_epochs=args.warmup_epochs, predictor_epochs=args.predictor_epochs,
        joint_epochs=args.joint_epochs, batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size, max_length=args.max_length,
        steps=args.steps, samples=args.samples, seed=args.seed, patience=args.patience,
        selection_scenes=tuple(args.selection_scenes), threads=args.threads,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train STIRDiff in three stages from raw CSVs")
    parser.add_argument("--train", required=True, type=Path, help="raw training CSV")
    parser.add_argument("--dev", required=True, type=Path, help="raw development CSV")
    parser.add_argument("--output", required=True, type=Path)
    add_training_arguments(parser)
    args = parser.parse_args(argv)
    train(_read_cli(args.train), _read_cli(args.dev), args.output,
          config_from_args(args), device=args.device)


if __name__ == "__main__":
    main()


__all__ = ["TrainingConfig", "train", "main"]
