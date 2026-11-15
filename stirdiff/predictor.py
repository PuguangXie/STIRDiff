"""STIRDiff risk prediction network."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn


@dataclass
class PredictorConfig:
    hidden_dim: int = 64
    heads: int = 4
    feature_layers: int = 2
    feedforward_dim: int = 128
    dropout: float = 0.1

    def __post_init__(self):
        if self.hidden_dim < 1 or self.heads < 1 or self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be positive and divisible by heads")
        if self.feature_layers < 1 or self.feedforward_dim < 1 or not 0 <= self.dropout < 1:
            raise ValueError("invalid model dimensions or dropout")


class _PreNormBlock(nn.Module):
    def __init__(self, d, h, ff, drop):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, d)
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x + self.drop(
            self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        )
        return x + self.drop(self.ff(self.norm2(x)))


class _RowEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_dim
        self.d = d
        self.scalar = nn.ModuleList([nn.Linear(1, d) for _ in range(9)])
        self.identity = nn.Parameter(torch.randn(9, d) * 0.02)
        self.mask_emb = nn.Parameter(torch.randn(6, d) * 0.02)
        self.gender_emb = nn.Embedding(3, d)
        self.scene_emb = nn.Embedding(6, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList([
            _PreNormBlock(d, config.heads, config.feedforward_dim, config.dropout)
            for _ in range(config.feature_layers)
        ])
        self.final_norm = nn.LayerNorm(d)

    def forward(self, vitals, observed, age, gender, scene, time):
        b, t, _ = vitals.shape
        values = torch.cat([vitals, age.unsqueeze(-1), time], -1)
        tokens = torch.stack([
            self.scalar[i](values[..., i:i + 1]) + self.identity[i] for i in range(9)
        ], 2).clone()
        tokens[:, :, :6] += observed.unsqueeze(-1) * self.mask_emb
        setting = self.scene_emb(scene)
        tokens = torch.cat([
            tokens, self.gender_emb(gender).unsqueeze(2), setting.unsqueeze(2)
        ], 2)
        query = self.cls.expand(b * t, -1, -1) + setting.reshape(b * t, 1, self.d)
        x = torch.cat([query, tokens.reshape(b * t, 11, self.d)], 1)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x[:, 0]).reshape(b, t, self.d), setting


class _ClockBranch(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, h = config.hidden_dim, config.heads
        self.heads, self.dh = h, d // h
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        base = torch.log(torch.tensor([15., 60., 240., 1440.]))
        if h > 4:
            base = torch.cat([base, base[-1].repeat(h - 4)])
        else:
            base = base[:h]
        self.log_tau = nn.Parameter(base.repeat(6, 1))

    def forward(self, row, scene, hospital_offset, valid):
        b, t, d = row.shape
        h, dh = self.heads, self.dh
        q = self.q(row).reshape(b, t, h, dh).transpose(1, 2)
        k0 = self.k(row).reshape(b, t, h, dh).transpose(1, 2)
        v0 = self.v(row).reshape(b, t, h, dh).transpose(1, 2)

        # A null key/value permits a finite result when no prior observation exists.
        k = torch.cat([torch.zeros_like(k0[:, :, :1]), k0], 2)
        v = torch.cat([torch.zeros_like(v0[:, :, :1]), v0], 2)
        key_valid = torch.cat([
            torch.ones(b, 1, dtype=torch.bool, device=row.device), valid
        ], 1)
        allowed = torch.cat([
            torch.ones(t, 1, dtype=torch.bool, device=row.device),
            torch.tril(torch.ones(t, t, dtype=torch.bool, device=row.device), diagonal=-1),
        ], 1)
        mask = ~(allowed.view(1, 1, t, t + 1) & key_valid[:, None, None, :])
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)

        current = hospital_offset.double()
        delta = (current.unsqueeze(-1) - current.unsqueeze(-2)).clamp_min(0).float()
        tau = (
            self.log_tau[scene.clamp(0, 5)].transpose(1, 2)
            .unsqueeze(-1).clamp(-8., 12.).exp()
        )
        scores = torch.cat([
            scores[..., :1], scores[..., 1:] - torch.log1p(delta.unsqueeze(1) / tau)
        ], -1)
        weights = torch.softmax(scores.masked_fill(mask, -torch.inf), -1)
        y = (weights @ v).transpose(1, 2).reshape(b, t, d)
        return self.out(y) * (valid & (torch.arange(t, device=row.device)[None] > 0)).unsqueeze(-1)


class RiskPredictor(nn.Module):
    """Accept prepared tensors and return one risk logit per observation.

    ``vitals`` and ``observed`` have shape [B,T,6]; ``time`` is [B,T,2].
    ``age``, ``gender``, ``scene``, ``hospital_offset`` and ``valid`` are [B,T].
    ``lengths`` is [B]. Sequences are ordered and right padded.
    """

    def __init__(self, config: PredictorConfig | None = None):
        super().__init__()
        self.config = config or PredictorConfig()
        d = self.config.hidden_dim
        self.risk_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.row_encoder = _RowEncoder(self.config)
        self.temporal = _ClockBranch(self.config)
        self.gate_net = nn.Linear(d + 2 + 6, 1)
        nn.init.zeros_(self.gate_net.weight)
        nn.init.constant_(self.gate_net.bias, math.log(0.1 / 0.9))
        self.float()

    def return_details(self, batch: Mapping[str, Tensor]):
        self._validate_batch(batch)
        observed = batch["observed"].float()
        valid = batch["valid"].bool()
        row, setting = self.row_encoder(
            batch["vitals"].float(), observed, batch["age"].float(),
            batch["gender"].long(), batch["scene"].long(), batch["time"].float(),
        )
        branch = self.temporal(row, batch["scene"].long(), batch["hospital_offset"], valid)
        gate = torch.sigmoid(self.gate_net(torch.cat([
            setting, batch["time"].float(), observed
        ], -1)))
        prior = (valid & (torch.arange(row.shape[1], device=row.device)[None] > 0)).unsqueeze(-1)
        gate = gate.masked_fill(~(valid.unsqueeze(-1) & prior), 0.)
        increment = gate * branch
        fused = row + increment
        logits = self.risk_head(fused).squeeze(-1).masked_fill(~valid, 0.)
        return {
            "row_repr": row, "temporal_increment": increment, "increment": increment,
            "gate": gate, "fused_repr": fused, "logits": logits,
        }

    def forward(self, batch: Mapping[str, Tensor]):
        return self.return_details(batch)["logits"]

    forward_details = return_details

    @staticmethod
    def _validate_batch(batch):
        required = {
            "vitals", "observed", "age", "gender", "scene", "time",
            "valid", "lengths", "hospital_offset",
        }
        missing = required - set(batch)
        if missing:
            raise ValueError(f"missing batch keys: {sorted(missing)}")
        b, t, f = batch["vitals"].shape
        if (f != 6 or batch["observed"].shape != (b, t, 6)
                or batch["time"].shape != (b, t, 2)
                or batch["hospital_offset"].shape != (b, t)):
            raise ValueError("invalid batch shapes")
        offset = batch["hospital_offset"].double()
        valid, lengths = batch["valid"].bool(), batch["lengths"].long()
        expected = torch.arange(t, device=lengths.device)[None] < lengths[:, None]
        if (lengths.numel() != b or (lengths < 1).any() or (lengths > t).any()
                or not torch.equal(valid.cpu(), expected.cpu())):
            raise ValueError("valid must be right-padded with at least one valid row")
        if not torch.isfinite(offset).all():
            raise ValueError("hospital_offset must be finite")
        if t > 1 and bool(((offset[:, 1:] - offset[:, :-1])[valid[:, 1:] & valid[:, :-1]] < 0).any()):
            raise ValueError("hospital_offset must be nondecreasing")


__all__ = ["PredictorConfig", "RiskPredictor"]
