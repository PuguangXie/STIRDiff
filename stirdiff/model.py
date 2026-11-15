"""Differentiable connection between imputation and risk prediction."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import os
import uuid
import numpy as np
import torch
from torch import nn
from .imputer import AxialDiffusionImputer, ImputerConfig
from .predictor import RiskPredictor, PredictorConfig

VITALS = ("heartrate", "sbp", "dbp", "resprate", "temperature", "spo2")
TIMES = ("vital_observationoffset", "scene_observationoffset")

class IntegratedModel(nn.Module):
    def __init__(self, generator_config, predictor_config, generator_preprocessor,
                 predictor_preprocessor, inference, freeze_generator=True):
        super().__init__()
        self.generator_preprocessor = deepcopy(generator_preprocessor)
        self.predictor_preprocessor = deepcopy(predictor_preprocessor)
        self.inference = deepcopy(inference)
        for pre in (self.generator_preprocessor, self.predictor_preprocessor):
            for key in (*VITALS, "age", *TIMES):
                if not np.isfinite(pre["means"][key]) or not np.isfinite(pre["scales"][key]) or pre["scales"][key] <= 0:
                    raise ValueError(f"Invalid preprocessing statistic: {key}")
        for key in ("steps", "samples", "max_length", "seed", *(
                ("predictor_max_length",) if "predictor_max_length" in self.inference else ())):
            v = self.inference[key]
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"{key} must be integer")
        if not 2 <= self.inference["steps"] <= generator_config["diffusion_steps"]:
            raise ValueError("Invalid diffusion sampling steps")
        if self.inference["samples"] < 1 or self.inference["max_length"] < 1 or self.prediction_max_length < 1 or self.inference["seed"] < 0:
            raise ValueError("Require positive samples/window lengths and nonnegative seed")
        if self.inference["aggregation"] not in {"mean", "median"}:
            raise ValueError("Aggregation must be mean or median")
        self.generator = AxialDiffusionImputer(ImputerConfig(**generator_config))
        self.predictor = RiskPredictor(PredictorConfig(**predictor_config))
        self.freeze_generator = bool(freeze_generator)
        self.generator.requires_grad_(not self.freeze_generator)
        if self.freeze_generator:
            self.generator.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_generator:
            self.generator.eval()
        return self

    @property
    def prediction_max_length(self):
        return self.inference.get("predictor_max_length", self.inference["max_length"])

    @staticmethod
    def _check(b, max_length):
        required = {"vitals", "observed", "age", "time", "gender", "scene",
                    "hospital_offset", "valid", "lengths", "window_id"}
        if required - set(b):
            raise ValueError(f"Missing input keys: {sorted(required - set(b))}")
        if b["vitals"].ndim != 3 or b["vitals"].shape[-1] != 6:
            raise ValueError("vitals must have shape [B,T,6]")
        n, t, _ = b["vitals"].shape
        if t < 1 or t > max_length:
            raise ValueError(f"Windows must contain 1..{max_length} rows for this saved component")
        shapes = dict(observed=(n,t,6), time=(n,t,2), age=(n,t), scene=(n,t),
                      gender=(n,t), hospital_offset=(n,t), valid=(n,t),
                      lengths=(n,), window_id=(n,t))
        if any(tuple(b[k].shape) != shape for k, shape in shapes.items()):
            raise ValueError("Invalid batch dimensions")
        for k in ("observed", "valid"):
            if not ((b[k] == 0) | (b[k] == 1)).all():
                raise ValueError(f"{k} must be binary")
        valid, lengths = b["valid"].bool(), b["lengths"]
        if (lengths < 1).any() or (lengths > t).any() or (lengths != lengths.long()).any():
            raise ValueError("Invalid window lengths")
        expected = torch.arange(t, device=lengths.device)[None] < lengths[:, None]
        if not torch.equal(valid, expected):
            raise ValueError("Require contiguous right padding")
        if not torch.isfinite(b["vitals"]).all() or not torch.isfinite(b["hospital_offset"]).all():
            raise ValueError("Use finite placeholders for missing vitals; offsets must be finite")
        if ((b["scene"][valid] < 1) | (b["scene"][valid] > 5) |
            (b["scene"][valid] != b["scene"][valid].long())).any():
            raise ValueError("Valid scenes must be integers 1..5")
        ids = b["window_id"][valid]
        if (ids < 0).any() or (ids != ids.long()).any():
            raise ValueError("window_id must be nonnegative integer")
        off = b["hospital_offset"].double()
        if t > 1 and ((off[:,1:] - off[:,:-1])[valid[:,1:]] < 0).any():
            raise ValueError("Offsets must be sorted within each window")

    @staticmethod
    def _norm(x, pre, keys, log=False):
        # CSV preprocessing originally computes in float64 before casting.
        x = x.double()
        if log:
            x = torch.log1p(x.clamp_min(0))
        means = x.new_tensor([pre["means"][k] for k in keys])
        scales = x.new_tensor([pre["scales"][k] for k in keys])
        return ((x - means) / scales).float()

    def _context(self, b, pre):
        age = b["age"].double()
        age = torch.where(torch.isfinite(age), age, age.new_tensor(pre["means"]["age"]))
        times = torch.where(torch.isfinite(b["time"]), b["time"], torch.zeros_like(b["time"]))
        gender = b["gender"]
        gender = torch.where((gender == 0) | (gender == 1) | (gender == 2),
                             gender, torch.full_like(gender, 2)).long()
        return dict(age=self._norm(age, pre, ("age",)),
                    time=self._norm(times, pre, TIMES, log=True),
                    gender=gender, scene=b["scene"].long(),
                    valid=b["valid"].bool(), lengths=b["lengths"].long(),
                    window_id=b["window_id"])

    def generator_batch(self, b):
        self._check(b, self.inference["max_length"])
        out = self._context(b, self.generator_preprocessor)
        mask = b["observed"].bool() & b["valid"].bool().unsqueeze(-1)
        out["vitals"] = torch.where(mask, self._norm(b["vitals"], self.generator_preprocessor, VITALS),
                                   torch.zeros_like(b["vitals"], dtype=torch.float32))
        out["vital_mask"] = mask.float()
        return out  # labels and patient identifiers do not enter the network

    def predictor_batch(self, b, physical_vitals):
        self._check(b, self.prediction_max_length)
        out = self._context(b, self.predictor_preprocessor)
        out["vitals"] = self._norm(physical_vitals, self.predictor_preprocessor, VITALS)
        out["observed"] = (b["observed"] * b["valid"].unsqueeze(-1)).float()
        out["hospital_offset"] = b["hospital_offset"].double()
        return out

    def _noise(self, batch, sample):
        noise = torch.zeros_like(batch["vitals"])
        for i, length in enumerate(batch["lengths"].cpu().tolist()):
            wid = int(batch["window_id"][i, 0])
            rng = np.random.default_rng(np.random.SeedSequence([self.inference["seed"], wid, sample]))
            values = rng.standard_normal((length, 6)).astype(np.float32)
            noise[i, :length] = torch.from_numpy(values).to(noise.device)
        return noise

    def complete(self, batch, gradient=True):
        gb = self.generator_batch(batch)
        was_training = self.generator.training
        self.generator.eval()  # sampling dropout is disabled for train AND test
        try:
            with torch.set_grad_enabled(bool(gradient) and torch.is_grad_enabled()):
                draws = torch.stack([self.generator.sample(
                    gb, steps=self.inference["steps"], initial_noise=self._noise(gb, s))
                    for s in range(self.inference["samples"])])
                x = draws.mean(0) if self.inference["aggregation"] == "mean" else torch.quantile(draws, .5, dim=0)
                pre = self.generator_preprocessor
                physical = x * x.new_tensor([pre["scales"][k] for k in VITALS])
                physical = physical + x.new_tensor([pre["means"][k] for k in VITALS])
                # Physical bounds apply only to imputed values.
                lo = x.new_tensor([0., 0., 0., 0., -273.15, 0.])
                hi = x.new_tensor([float("inf")]*5 + [100.])
                physical = torch.maximum(torch.minimum(physical, hi), lo)
                result = torch.where(batch["observed"].bool(), batch["vitals"], physical)
                return result.masked_fill(~batch["valid"].unsqueeze(-1), 0.)
        finally:
            self.generator.train(was_training and not self.freeze_generator)

    def forward(self, batch, input_mode="raw"):
        if input_mode not in {"raw", "preimputed"}:
            raise ValueError("Unknown input mode")
        physical = batch["vitals"] if input_mode == "preimputed" else self.complete(
            batch, gradient=not self.freeze_generator)
        return self.predictor(self.predictor_batch(batch, physical))

    def diffusion_loss(self, batch, cond_mask, target_mask):
        return self.generator.loss(self.generator_batch(batch), cond_mask, target_mask)

    def config(self):
        return dict(generator_config=asdict(self.generator.config),
                    predictor_config=asdict(self.predictor.config),
                    generator_preprocessor=deepcopy(self.generator_preprocessor),
                    predictor_preprocessor=deepcopy(self.predictor_preprocessor),
                    inference=deepcopy(self.inference), freeze_generator=self.freeze_generator)


def save_checkpoint(path, model, threshold=None):
    """Save model state and its development-set Youden cutoff, if available.

    A cutoff must be selected on development predictions for these exact
    weights. The imputer-only warmup checkpoint has no classification cutoff.
    """
    if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("Threshold must be finite and within [0, 1]")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    payload = {
        "format_version": 1,
        "config": model.config(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "threshold": threshold,
        "threshold_selection": None if threshold is None else {
            "data": "development", "criterion": "youden_j",
            "decision_rule": "probability >= threshold",
        },
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, device="cpu"):
    """Load a checkpoint written by save_checkpoint; return model and cutoff."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    model = IntegratedModel(**saved["config"])
    model.load_state_dict(saved["state_dict"], strict=True)
    model.to(device).eval()
    threshold = saved.get("threshold")
    if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("Invalid saved threshold")
    selection = saved.get("threshold_selection")
    if selection is not None and selection != {
            "data": "development", "criterion": "youden_j",
            "decision_rule": "probability >= threshold"}:
        raise ValueError("Checkpoint cutoff must come from development-set Youden J")
    return model, threshold
