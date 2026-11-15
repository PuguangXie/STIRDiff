"""Feature-time axial diffusion imputer with zero-terminal-SNR v prediction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class ImputerConfig:
    """Network dimensions for the feature-time architecture."""

    hidden_dim: int = 64
    layers: int = 2
    heads: int = 4
    feedforward_dim: int = 128
    dropout: float = 0.1
    diffusion_steps: int = 100
    x0_huber_weight: float = 0.05


def _sinusoidal_embedding(steps: Tensor, dim: int) -> Tensor:
    half = dim // 2
    if half == 0:
        return steps[:, None]
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=steps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = steps.float()[:, None] * frequencies[None, :]
    result = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if result.shape[-1] < dim:
        result = F.pad(result, (0, dim - result.shape[-1]))
    return result


class _AxialBlock(nn.Module):
    """Within-row feature attention and causal temporal attention."""

    def __init__(self, hidden_dim: int, heads: int, feedforward_dim: int, dropout: float) -> None:
        super().__init__()
        self.feature_norm = nn.LayerNorm(hidden_dim)
        self.feature_attn = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.temporal_attn = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.temporal_amplitude_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, valid: Tensor, causal_mask: Tensor) -> Tensor:
        # x is [B, T, V, H]. Feature attention is independent in every row.
        b, t, v, h = x.shape
        y = self.feature_norm(x).reshape(b * t, v, h)
        y, _ = self.feature_attn(y, y, y, need_weights=False)
        x = x + self.dropout(y.reshape(b, t, v, h))

        # Strict-past attention. Row zero's temporal residual is zeroed below.
        y = self.temporal_norm(x).permute(0, 2, 1, 3).reshape(b * v, t, h)
        key_padding = (~valid).unsqueeze(1).expand(b, v, t).reshape(b * v, t)
        y, _ = self.temporal_attn(
            y, y, y, attn_mask=causal_mask, key_padding_mask=key_padding,
            need_weights=False,
        )
        y = y.reshape(b, v, t, h).permute(0, 2, 1, 3)
        temporal_gate = valid[:, :, None, None].to(y.dtype).clone()
        temporal_gate[:, 0] = 0.0
        x = x + self.dropout(y) * torch.sigmoid(self.temporal_amplitude_logit) * temporal_gate
        return x + self.ff(self.ff_norm(x))


class AxialDiffusionImputer(nn.Module):
    """Causal feature-time denoiser and DDIM imputer for six variables."""

    N_VARIABLES = 6

    def __init__(self, config: ImputerConfig) -> None:
        super().__init__()
        self.config = config
        integer_fields = ("hidden_dim", "layers", "heads", "feedforward_dim", "diffusion_steps")
        if any(isinstance(getattr(config, field), bool) or not isinstance(getattr(config, field), int)
               for field in integer_fields):
            raise ValueError("hidden_dim, layers, heads, feedforward_dim and diffusion_steps must be integers")
        if config.hidden_dim <= 0 or config.layers <= 0 or config.heads <= 0 or config.feedforward_dim <= 0:
            raise ValueError("hidden_dim, layers, heads and feedforward_dim must be positive")
        if config.hidden_dim % config.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if config.diffusion_steps < 2:
            raise ValueError("diffusion_steps must be at least 2")
        if not (0.0 <= float(config.dropout) < 1.0):
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if float(config.x0_huber_weight) < 0.0:
            raise ValueError("x0_huber_weight must be nonnegative")

        # Cosine schedule rescaled to exact zero terminal SNR.
        d = int(config.diffusion_steps)
        positions = torch.arange(d + 1, dtype=torch.float64) / d
        s = 0.008
        raw = torch.cos((positions + s) / (1.0 + s) * math.pi / 2.0).square()
        raw = raw / raw[0]
        root = raw.sqrt()
        root = (root - root[-1]) / (root[0] - root[-1])
        raw = root.square()
        raw[0] = 1.0
        raw[-1] = 0.0
        self.register_buffer("alpha_bar", raw.float(), persistent=True)
        self.register_buffer("sqrt_alpha_bar", raw.clamp_min(0).sqrt().float(), persistent=True)
        self.register_buffer(
            "sqrt_one_minus_alpha_bar", (1.0 - raw).clamp_min(0).sqrt().float(), persistent=True
        )
        causal = torch.triu(torch.ones(1, 1), diagonal=0).bool()
        self.register_buffer("_causal_placeholder", causal, persistent=False)

        h = config.hidden_dim
        self.value_weight = nn.Parameter(torch.randn(self.N_VARIABLES, h) / math.sqrt(h))
        self.value_bias = nn.Parameter(torch.zeros(self.N_VARIABLES, h))
        self.conditional_weight = nn.Parameter(torch.randn(self.N_VARIABLES, h) / math.sqrt(h))
        self.conditional_bias = nn.Parameter(torch.zeros(self.N_VARIABLES, h))
        self.mask_projection = nn.Linear(1, h)
        self.variable_embedding = nn.Embedding(self.N_VARIABLES, h)
        self.scene_embedding = nn.Embedding(6, h)
        self.gender_embedding = nn.Embedding(3, h)
        self.age_projection = nn.Linear(1, h)
        self.time_projection = nn.Linear(2, h)
        self.diffusion_projection = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
        self.blocks = nn.ModuleList(
            _AxialBlock(h, config.heads, config.feedforward_dim, config.dropout)
            for _ in range(config.layers)
        )
        self.output_norm = nn.LayerNorm(h)
        self.output_projection = nn.Linear(h, 1)

    def _causal_mask(self, length: int, device: torch.device) -> Tensor:
        mask = torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=0)
        if length:
            mask[0, 0] = False
        return mask

    @staticmethod
    def _binary_mask(mask: Tensor, name: str, shape: tuple[int, ...]) -> Tensor:
        if tuple(mask.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if mask.dtype.is_floating_point and not torch.isfinite(mask).all():
            raise ValueError(f"{name} must contain finite binary values")
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError(f"{name} must contain only 0/1 values")
        return mask.bool()

    def _valid_mask(self, batch: Mapping[str, Tensor], b: int, t: int) -> Tensor:
        valid = self._binary_mask(batch["valid"], "valid", (b, t))
        if t > 1 and bool(((~valid[:, :-1]) & valid[:, 1:]).any()):
            raise ValueError("valid must be contiguous and right padded")
        return valid

    def _validate_masks(
        self,
        batch: Mapping[str, Tensor],
        cond_mask: Tensor,
        target_mask: Tensor | None = None,
        *,
        allow_default_padding_mask: bool = False,
    ) -> tuple[Tensor, Tensor | None, Tensor, Tensor]:
        b, t, _ = batch["vitals"].shape
        valid = self._valid_mask(batch, b, t)
        original = self._binary_mask(batch["vital_mask"], "vital_mask", (b, t, self.N_VARIABLES))
        cond = self._binary_mask(cond_mask, "cond_mask", (b, t, self.N_VARIABLES))
        if allow_default_padding_mask:
            cond = cond & valid.unsqueeze(-1)
        elif bool((cond & ~valid.unsqueeze(-1)).any()):
            raise ValueError("cond_mask cannot include padded positions")
        if bool((cond & ~original).any()):
            raise ValueError("cond_mask cannot include originally missing cells")
        target = None
        if target_mask is not None:
            target = self._binary_mask(target_mask, "target_mask", (b, t, self.N_VARIABLES))
            if bool((target & ~valid.unsqueeze(-1)).any()):
                raise ValueError("target_mask cannot include padded positions")
            if bool((target & ~original).any()):
                raise ValueError("target_mask cannot include originally missing cells")
            if bool((target & cond).any()):
                raise ValueError("target_mask and cond_mask must be disjoint")
        return cond, target, original, valid

    def _check_batch(self, batch: Mapping[str, Tensor], noisy: Tensor | None = None) -> tuple[int, int, torch.device]:
        required = ("vitals", "scene", "age", "gender", "time", "valid")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"batch missing required key(s): {missing}")
        values = batch["vitals"] if noisy is None else noisy
        if values.ndim != 3 or values.shape[-1] != self.N_VARIABLES:
            raise ValueError("vitals/noisy must have shape [B,T,6]")
        b, t, _ = values.shape
        if tuple(batch["scene"].shape) != (b, t) or tuple(batch["age"].shape) != (b, t):
            raise ValueError("scene and age must have shape [B,T]")
        if tuple(batch["gender"].shape) != (b, t) or tuple(batch["valid"].shape) != (b, t):
            raise ValueError("gender and valid must have shape [B,T]")
        if tuple(batch["time"].shape) != (b, t, 2):
            raise ValueError("time must have shape [B,T,2]")
        return b, t, values.device

    def denoise(self, noisy: Tensor, cond_mask: Tensor, batch: Mapping[str, Tensor], step: Tensor) -> Tensor:
        """Predict the six-channel v target from noisy values and context."""
        b, t, device = self._check_batch(batch, noisy)
        cond_bool = self._binary_mask(cond_mask, "cond_mask", (b, t, self.N_VARIABLES))
        if step.ndim == 0:
            step = step.expand(b)
        if tuple(step.shape) != (b,):
            raise ValueError("step must have shape [B]")
        cond = cond_bool.to(dtype=noisy.dtype)
        clean_values = batch["vitals"].to(dtype=noisy.dtype)
        clean = torch.where(cond_bool, clean_values, torch.zeros_like(clean_values))
        x = (
            noisy.unsqueeze(-1) * self.value_weight[None, None, :, :] + self.value_bias[None, None, :, :]
            + clean.unsqueeze(-1) * self.conditional_weight[None, None, :, :] + self.conditional_bias[None, None, :, :]
            + self.mask_projection(cond.unsqueeze(-1))
        )
        var = self.variable_embedding(torch.arange(self.N_VARIABLES, device=device))[None, None]
        scene = batch["scene"].long().clamp(0, 5)
        gender = batch["gender"].long().clamp(0, 2)
        context = (
            self.scene_embedding(scene) + self.gender_embedding(gender)
            + self.age_projection(batch["age"].to(noisy.dtype).unsqueeze(-1))
            + self.time_projection(batch["time"].to(noisy.dtype))
        )
        step_context = self.diffusion_projection(_sinusoidal_embedding(step.to(device), self.config.hidden_dim))
        x = x + var + context.unsqueeze(2) + step_context[:, None, None, :]
        valid = self._valid_mask(batch, b, t)
        causal = self._causal_mask(t, device)
        for block in self.blocks:
            x = block(x, valid, causal)
        result = self.output_projection(self.output_norm(x)).squeeze(-1)
        return result.masked_fill(~valid.unsqueeze(-1), 0.0)

    def _step_coefficients(self, step: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        step = step.long().clamp(0, self.config.diffusion_steps)
        a = self.alpha_bar.to(dtype=dtype, device=step.device).index_select(0, step)
        return a.sqrt(), (1.0 - a).clamp_min(0).sqrt()

    def _predict_x0_eps(self, noisy: Tensor, prediction: Tensor, step: Tensor) -> tuple[Tensor, Tensor]:
        sa, so = self._step_coefficients(step, noisy.dtype)
        while sa.ndim < noisy.ndim:
            sa = sa.unsqueeze(-1)
            so = so.unsqueeze(-1)
        x0 = sa * noisy - so * prediction
        eps = so * noisy + sa * prediction
        return x0, eps

    def loss(
        self,
        batch: Mapping[str, Tensor],
        cond_mask: Tensor,
        target_mask: Tensor,
        loss_weights: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Compute scene-balanced masked v-prediction MSE and x0 SmoothL1."""
        b, t, device = self._check_batch(batch)
        vitals = batch["vitals"]
        dtype = vitals.dtype if vitals.is_floating_point() else torch.float32
        vitals = vitals.to(dtype)
        if "vital_mask" not in batch:
            raise KeyError("loss requires original batch['vital_mask'] for target validation")
        cond_mask, target_mask, original, valid_2d = self._validate_masks(
            batch, cond_mask, target_mask
        )
        assert target_mask is not None
        support = target_mask & valid_2d.unsqueeze(-1)
        if loss_weights is None and "loss_weight" in batch:
            loss_weights = batch["loss_weight"]
        if loss_weights is None:
            weights = torch.ones_like(vitals)
        else:
            if tuple(loss_weights.shape) != (b, t, self.N_VARIABLES):
                raise ValueError("loss_weights must have shape [B,T,6]")
            weights = loss_weights.to(device=device, dtype=dtype)
            if not torch.isfinite(weights[support]).all() or bool((weights[support] <= 0).any()):
                raise ValueError("loss weights on target cells must be finite and positive")
            weights = torch.where(support, weights, torch.ones_like(weights))
        count = int(support.sum().item())
        if count == 0:
            zero = sum((parameter.sum() for parameter in self.parameters()), vitals.new_zeros(())) * 0.0
            return zero, {"target_count": 0, "support_count": 0, "groups": 0, "loss": 0.0, "finite": True}

        step = torch.randint(1, self.config.diffusion_steps + 1, (b,), device=device)
        sa, so = self._step_coefficients(step, dtype)
        sa3, so3 = sa[:, None, None], so[:, None, None]
        original3 = original.to(device=device)
        latent_clean = torch.where(original3, vitals, torch.randn_like(vitals))
        noise = torch.randn_like(latent_clean)
        noisy = sa3 * latent_clean + so3 * noise
        denoise_input = torch.where(cond_mask, vitals, noisy)
        prediction = self.denoise(denoise_input, cond_mask, batch, step)
        target = sa3 * noise - so3 * latent_clean
        per_cell = (prediction - target).square()

        scene = batch["scene"].long()
        group_losses: list[Tensor] = []
        group_support = 0
        for scene_id in range(1, 6):
            for variable in range(self.N_VARIABLES):
                group = support[..., variable] & (scene == scene_id)
                if group.any():
                    group_weights = weights[..., variable][group]
                    group_losses.append(
                        (per_cell[..., variable][group] * group_weights).sum()
                        / group_weights.sum()
                    )
                    group_support += 1
        if not group_losses:
            raise RuntimeError("target support unexpectedly vanished")
        diffusion_loss = torch.stack(group_losses).mean()

        x0, _ = self._predict_x0_eps(denoise_input, prediction, step)
        aux_groups: list[Tensor] = []
        for scene_id in range(1, 6):
            for variable in range(self.N_VARIABLES):
                group = support[..., variable] & (scene == scene_id)
                if group.any():
                    group_weights = weights[..., variable][group]
                    aux_values = F.smooth_l1_loss(
                        x0[..., variable][group], latent_clean[..., variable][group], reduction="none"
                    )
                    aux_groups.append(
                        (aux_values * group_weights).sum() / group_weights.sum()
                    )
        aux = torch.stack(aux_groups).mean()
        total = diffusion_loss + float(self.config.x0_huber_weight) * aux
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite diffusion loss")
        stats = {
            "target_count": count,
            "support_count": count,
            "groups": group_support,
            "diffusion_loss": float(diffusion_loss.detach().item()),
            "x0_loss": float(aux.detach().item()),
            "loss": float(total.detach().item()),
            "finite": True,
        }
        return total, stats

    def _randn(self, shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype,
               generator: torch.Generator | None = None) -> Tensor:
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)

    def sample(
        self,
        batch: Mapping[str, Tensor],
        cond_mask: Tensor | None = None,
        steps: int = 6,
        initial_noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Run deterministic DDIM (eta=0), clamping observed conditions."""
        b, t, device = self._check_batch(batch)
        if isinstance(steps, bool) or not isinstance(steps, int) or not (2 <= steps <= self.config.diffusion_steps):
            raise ValueError("steps must satisfy 2 <= steps <= diffusion_steps")
        if "vital_mask" not in batch:
            raise KeyError("sample requires batch['vital_mask'] for condition validation")
        if cond_mask is None:
            original = self._binary_mask(batch["vital_mask"], "vital_mask", (b, t, self.N_VARIABLES))
            cond_mask = original & batch["valid"].bool().unsqueeze(-1)
        if tuple(cond_mask.shape) != (b, t, self.N_VARIABLES):
            raise ValueError(f"cond_mask must have shape {(b, t, self.N_VARIABLES)}")
        cond = self._binary_mask(cond_mask, "cond_mask", (b, t, self.N_VARIABLES))
        original = self._binary_mask(batch["vital_mask"], "vital_mask", (b, t, self.N_VARIABLES))
        valid = self._valid_mask(batch, b, t)
        if bool((cond & ~valid.unsqueeze(-1)).any()) or bool((cond & ~original).any()):
            raise ValueError("cond_mask must select originally observed, valid positions")
        dtype = batch["vitals"].dtype if batch["vitals"].is_floating_point() else torch.float32
        clean = batch["vitals"].to(device=device, dtype=dtype)
        x = self._randn((b, t, 6), device=device, dtype=dtype, generator=generator) if initial_noise is None else initial_noise.to(device=device, dtype=dtype).clone()
        if tuple(x.shape) != (b, t, 6):
            raise ValueError("initial_noise must have shape [B,T,6]")
        x = torch.where(cond, clean, x).masked_fill(~valid.unsqueeze(-1), 0.0)
        schedule = torch.linspace(self.config.diffusion_steps, 1, steps, device=device).round().long()
        for index in range(schedule.numel()):
            current = schedule[index].expand(b)
            previous_value = schedule[index + 1] if index + 1 < schedule.numel() else 0
            previous = torch.as_tensor(previous_value, device=device, dtype=torch.long).expand(b)
            prediction = self.denoise(x, cond, batch, current)
            x0, eps = self._predict_x0_eps(x, prediction, current)
            sa_prev, so_prev = self._step_coefficients(previous, dtype)
            x = sa_prev[:, None, None] * x0 + so_prev[:, None, None] * eps
            x = torch.where(cond, clean, x).masked_fill(~valid.unsqueeze(-1), 0.0)
        x = torch.where(cond, clean, x)
        return x.masked_fill(~valid.unsqueeze(-1), 0.0)

    def impute(
        self,
        batch: Mapping[str, Tensor],
        steps: int = 6,
        samples: int = 5,
        aggregation: str = "mean",
        generator: torch.Generator | None = None,
        return_samples: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Generate multiple imputations and aggregate, preserving observations."""
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise ValueError("samples must be a positive integer")
        aggregation = aggregation.lower()
        if aggregation not in {"mean", "median"}:
            raise ValueError("aggregation must be 'mean' or 'median'")
        if "vital_mask" not in batch:
            raise KeyError("impute requires batch['vital_mask']")
        b, t, _ = batch["vitals"].shape
        original = self._binary_mask(batch["vital_mask"], "vital_mask", (b, t, self.N_VARIABLES))
        valid = self._valid_mask(batch, b, t)
        cond = original & valid.unsqueeze(-1)
        draws = torch.stack(
            [self.sample(batch, cond_mask=cond, steps=steps, generator=generator) for _ in range(samples)], dim=0
        )
        result = draws.mean(dim=0) if aggregation == "mean" else torch.quantile(draws, 0.5, dim=0)
        result = torch.where(cond, batch["vitals"].to(result.dtype), result)
        result = result.masked_fill(~valid.unsqueeze(-1), 0.0)
        return (result, draws) if return_samples else result


__all__ = ["ImputerConfig", "AxialDiffusionImputer"]
