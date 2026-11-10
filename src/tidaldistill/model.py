# -*- coding: utf-8 -*-
"""Inference-time implementation of TidalDistill."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


FEATURE_COLUMNS = [
    "sex", "age", "height_cm", "weight_kg", "bmi", "label",
    "rf", "hr", "power", "p_syst", "p_diast", "hrr", "spo2",
    "rf_rate", "hr_rate", "power_rate", "hrr_rate",
]
STATIC_FEATURES = ["sex", "age", "height_cm", "weight_kg", "bmi", "label"]
DYNAMIC_FEATURES = [
    "rf", "hr", "power", "p_syst", "p_diast", "hrr", "spo2",
    "rf_rate", "hr_rate", "power_rate", "hrr_rate",
]
STATIC_CATEGORICAL = ["sex", "label"]
CATEGORY_COUNTS = {"sex": 3, "label": 6}
WINDOW_SIZE = 32


class GatedResidualNetwork(nn.Module):
    """Small GRN block used for static conditioning and token fusion."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int | None = None,
        context_size: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        output_size = output_size or input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.context = nn.Linear(context_size, hidden_size, bias=False) if context_size is not None else None
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(output_size, output_size)
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        residual = self.skip(x)
        hidden = self.fc1(x)
        if context is not None and self.context is not None:
            context_hidden = self.context(context)
            while context_hidden.dim() < hidden.dim():
                context_hidden = context_hidden.unsqueeze(1)
            hidden = hidden + context_hidden
        hidden = F.elu(hidden)
        hidden = self.dropout(self.fc2(hidden))
        gated = torch.sigmoid(self.gate(hidden)) * hidden
        return self.norm(residual + gated)


class CausalConvBlock(nn.Module):
    """Length-preserving causal Conv1d block for [B, T, D] tensors."""

    def __init__(self, channels: int, kernel_size: int, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.pad = nn.ConstantPad1d(((kernel_size - 1) * dilation, 0), 0.0)
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        y = self.conv(self.pad(y))
        y = self.dropout(F.gelu(self.norm(y)))
        return y.transpose(1, 2)


class StaticConditionalEncoder(nn.Module):
    """Categorical embedding + continuous projection -> subject condition c."""

    def __init__(
        self,
        feature_to_idx: Mapping[str, int],
        static_categorical: Sequence[str],
        static_continuous: Sequence[str],
        category_counts: Mapping[str, int],
        embedding_dim: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.feature_to_idx = dict(feature_to_idx)
        self.static_categorical = list(static_categorical)
        self.static_continuous = list(static_continuous)
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(max(int(category_counts.get(name, 1)), 1), embedding_dim)
                for name in self.static_categorical
            }
        )
        input_size = len(self.static_continuous) + len(self.static_categorical) * embedding_dim
        self.encoder = GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        static_step = x[:, -1, :]
        parts: List[torch.Tensor] = []
        if self.static_continuous:
            indices = [self.feature_to_idx[name] for name in self.static_continuous]
            parts.append(static_step[:, indices])
        for name in self.static_categorical:
            values = static_step[:, self.feature_to_idx[name]].round().long()
            values = values.clamp(0, self.embeddings[name].num_embeddings - 1)
            parts.append(self.embeddings[name](values))
        if not parts:
            return static_step.new_zeros(static_step.shape[0], self.encoder.fc2.out_features)
        return self.encoder(torch.cat(parts, dim=-1))





class VariableRelationStream(nn.Module):
    """Student-side K-by-K variable relation stream."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        heads = num_heads if hidden_size % num_heads == 0 else 1
        self.attention = nn.MultiheadAttention(hidden_size, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.ffn = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        variable_tokens = tokens.mean(dim=1)
        variable_tokens = variable_tokens + context.unsqueeze(1)
        attended, relation_map = self.attention(
            variable_tokens,
            variable_tokens,
            variable_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        variable_tokens = self.ffn(self.norm(variable_tokens + attended))
        patch_context = variable_tokens.unsqueeze(1).expand(-1, tokens.shape[1], -1, -1).mean(dim=2)
        return patch_context, variable_tokens, relation_map


class HierarchicalTrendDecoder(nn.Module):
    """Coarse-to-fine trend reconstruction decoder."""

    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.smoother = nn.Sequential(
            CausalConvBlock(hidden_size, kernel_size=9, dilation=1, dropout=dropout),
            CausalConvBlock(hidden_size, kernel_size=7, dilation=2, dropout=dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, fused_sequence: torch.Tensor) -> torch.Tensor:
        return self.smoother(fused_sequence).squeeze(-1)


class ResidualSpectralDecoder(nn.Module):
    """Local residual reconstruction plus spectral output."""

    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.local = nn.Sequential(
            CausalConvBlock(hidden_size, kernel_size=3, dilation=1, dropout=dropout),
            CausalConvBlock(hidden_size, kernel_size=3, dilation=2, dropout=dropout),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, fused_sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.local(fused_sequence).squeeze(-1)
        spectrum = torch.abs(torch.fft.rfft(residual, dim=1))
        return residual, spectrum


class RelationAwareAdaptiveFusion(nn.Module):
    """Adaptive residual injection: pred = trend + u(t) * residual."""

    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2 + 2, hidden_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        fused_sequence: torch.Tensor,
        context: torch.Tensor,
        trend_seq: torch.Tensor,
        residual_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context_seq = context.unsqueeze(1).expand(-1, fused_sequence.shape[1], -1)
        gate_input = torch.cat(
            [fused_sequence, context_seq, trend_seq.unsqueeze(-1), residual_seq.unsqueeze(-1)],
            dim=-1,
        )
        residual_gate = torch.sigmoid(self.gate(gate_input)).squeeze(-1)
        return trend_seq + residual_gate * residual_seq, residual_gate


class VariableTransformerBlock(nn.Module):
    """iTransformer-style cross-variable encoder block."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        heads = num_heads if hidden_size % num_heads == 0 else 1
        self.attention = nn.MultiheadAttention(hidden_size, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = GatedResidualNetwork(hidden_size, hidden_size * 2, hidden_size, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attended, attention = self.attention(tokens, tokens, tokens, need_weights=True, average_attn_weights=False)
        tokens = self.norm1(tokens + attended)
        tokens = self.norm2(tokens + self.ffn(tokens))
        return tokens, attention


class TemporalSelfAttentionBlock(nn.Module):
    """Small temporal attention refinement block for [B, T, D] sequences."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        heads = num_heads if hidden_size % num_heads == 0 else 1
        self.attention = nn.MultiheadAttention(hidden_size, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = GatedResidualNetwork(hidden_size, hidden_size * 2, hidden_size, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attended, attention = self.attention(sequence, sequence, sequence, need_weights=True, average_attn_weights=False)
        sequence = self.norm1(sequence + attended)
        sequence = self.norm2(sequence + self.ffn(sequence))
        return sequence, attention


class DualFeatureSelection(nn.Module):
    """Static-context-conditioned trend/detail variable weighting."""

    def __init__(self, hidden_size: int, dropout: float, mode: str = "dual") -> None:
        super().__init__()
        self.mode = str(mode).lower()
        if self.mode not in {"dual", "shared", "uniform"}:
            raise ValueError(f"Unknown feature-selection mode: {mode}")
        self.trend_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.detail_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.shared_gate = (
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )
            if self.mode == "shared"
            else None
        )

    def forward(self, embeddings: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context_tokens = context.view(context.shape[0], 1, 1, -1).expand(-1, embeddings.shape[1], embeddings.shape[2], -1)
        gate_input = torch.cat([embeddings, context_tokens], dim=-1)
        if self.mode == "uniform":
            trend_weights = embeddings.new_full(embeddings.shape[:3], 1.0 / max(embeddings.shape[2], 1))
            detail_weights = trend_weights
        elif self.mode == "shared":
            assert self.shared_gate is not None
            trend_weights = torch.softmax(self.shared_gate(gate_input).squeeze(-1), dim=-1)
            detail_weights = trend_weights
        else:
            trend_weights = torch.softmax(self.trend_gate(gate_input).squeeze(-1), dim=-1)
            detail_weights = torch.softmax(self.detail_gate(gate_input).squeeze(-1), dim=-1)
        trend_sequence = (trend_weights.unsqueeze(-1) * embeddings).sum(dim=2)
        detail_sequence = (detail_weights.unsqueeze(-1) * embeddings).sum(dim=2)
        return trend_sequence, detail_sequence, trend_weights, detail_weights


class TFLSTMExtractor(nn.Module):
    """TFN-style LSTM lane with temporal self-attention."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.attention_block = TemporalSelfAttentionBlock(hidden_size, num_heads, dropout)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, _ = self.lstm(sequence)
        encoded = self.norm(encoded + sequence)
        return self.attention_block(encoded)


class TimeFusionPredictionHeads(nn.Module):
    """Student trend/residual heads with an adaptive residual gate."""

    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.trend_head = HierarchicalTrendDecoder(hidden_size, dropout)
        self.residual_head = ResidualSpectralDecoder(hidden_size, dropout)
        self.detail_gate = RelationAwareAdaptiveFusion(hidden_size, dropout)

    def forward(self, sequence: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        trend_seq = self.trend_head(sequence)
        residual_seq, residual_spectrum = self.residual_head(sequence)
        pred_seq, gate = self.detail_gate(sequence, context, trend_seq, residual_seq)
        return trend_seq, residual_seq, residual_spectrum, pred_seq, gate


class PhysiologicalPointEmbedding(nn.Module):
    """Point-wise physiological tokenization for dynamic CPET values [B, T, K]."""

    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        feature_id_dim: int,
        dropout: float,
        use_feature_identity: bool = True,
        max_positions: int = 512,
        num_stages: int | None = None,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.hidden_size = int(hidden_size)
        self.use_feature_identity = bool(use_feature_identity)
        self.value_projection = nn.Linear(2, hidden_size)
        self.position_embedding = nn.Embedding(max_positions, hidden_size)
        harmonic_dim = 8
        self.harmonic_projection = nn.Linear(harmonic_dim, hidden_size)
        self.feature_identity = nn.Embedding(num_features, max(int(feature_id_dim), 1))
        self.feature_identity_projection = nn.Linear(max(int(feature_id_dim), 1), hidden_size)
        self.stage_embedding = nn.Embedding(num_stages, hidden_size) if num_stages is not None else None
        self.mask_projection = nn.Linear(1, hidden_size)
        self.time_since_projection = nn.Linear(1, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.max_positions = int(max_positions)

    def _harmonic_time_embedding(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if length <= 1:
            positions = torch.zeros((length,), device=device, dtype=dtype)
        else:
            positions = torch.arange(length, device=device, dtype=dtype) / float(length - 1)
        frequencies = torch.tensor([1.0, 2.0, 4.0, 8.0], device=device, dtype=dtype)
        angles = 2.0 * torch.pi * positions.unsqueeze(-1) * frequencies.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(
        self,
        x_dynamic: torch.Tensor,
        stage: torch.Tensor | None = None,
        observation_mask: torch.Tensor | None = None,
        time_since_last_observation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        delta = torch.zeros_like(x_dynamic)
        if x_dynamic.shape[1] > 1:
            delta[:, 1:] = x_dynamic[:, 1:] - x_dynamic[:, :-1]

        tokens = self.value_projection(torch.stack([x_dynamic, delta], dim=-1))
        positions = torch.arange(x_dynamic.shape[1], device=x_dynamic.device).clamp(max=self.max_positions - 1)
        tokens = tokens + self.position_embedding(positions).view(1, x_dynamic.shape[1], 1, -1)
        harmonic = self._harmonic_time_embedding(x_dynamic.shape[1], x_dynamic.device, x_dynamic.dtype)
        tokens = tokens + self.harmonic_projection(harmonic).view(1, x_dynamic.shape[1], 1, -1)

        if self.use_feature_identity:
            feature_ids = torch.arange(x_dynamic.shape[2], device=x_dynamic.device)
            feature_tokens = self.feature_identity_projection(self.feature_identity(feature_ids))
            tokens = tokens + feature_tokens.view(1, 1, x_dynamic.shape[2], -1)

        if stage is not None and self.stage_embedding is not None:
            stage_values = stage.long().clamp(0, self.stage_embedding.num_embeddings - 1)
            stage_tokens = self.stage_embedding(stage_values)
            while stage_tokens.dim() < tokens.dim():
                stage_tokens = stage_tokens.unsqueeze(-2)
            tokens = tokens + stage_tokens

        if observation_mask is not None:
            tokens = tokens + self.mask_projection(observation_mask.float().unsqueeze(-1))
        if time_since_last_observation is not None:
            tokens = tokens + self.time_since_projection(time_since_last_observation.float().unsqueeze(-1))

        return self.dropout(self.norm(tokens))


class _LocalTemporalBranch(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = ((max(int(kernel_size), 1) - 1) * max(int(dilation), 1)) // 2
        self.depthwise = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=max(int(kernel_size), 1),
            dilation=max(int(dilation), 1),
            padding=padding,
            groups=hidden_size,
        )
        self.pointwise = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.sci_even = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, groups=hidden_size)
        self.sci_odd = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, groups=hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def _sci_interaction(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] <= 2:
            return values
        even = values[..., 0::2]
        odd = values[..., 1::2]
        if odd.shape[-1] < even.shape[-1]:
            odd_for_even = F.pad(odd, (0, even.shape[-1] - odd.shape[-1]), mode="replicate")
        else:
            odd_for_even = odd[..., : even.shape[-1]]
        even_update = even + 0.1 * torch.tanh(self.sci_even(odd_for_even))
        odd_update = odd + 0.1 * torch.tanh(self.sci_odd(even[..., : odd.shape[-1]]))
        out = values.clone()
        out[..., 0::2] = even_update
        out[..., 1::2] = odd_update
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.depthwise(x)
        y = F.gelu(y)
        y = self.dropout(self.pointwise(y))
        y = self._sci_interaction(y)
        return self.norm((residual + y).transpose(1, 2)).transpose(1, 2)


class MultiScaleLocalTemporalEncoder(nn.Module):
    """Per-variable multi-scale local temporal encoder for [B, T, K, D]."""

    def __init__(
        self,
        hidden_size: int,
        kernels: Sequence[int] = (3, 5, 9),
        dilations: Sequence[int] = (1, 1, 2),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        kernel_values = list(kernels)[:3]
        dilation_values = list(dilations)[:3]
        while len(kernel_values) < 3:
            kernel_values.append(kernel_values[-1] if kernel_values else 3)
        while len(dilation_values) < 3:
            dilation_values.append(dilation_values[-1] if dilation_values else 1)
        self.branches = nn.ModuleList(
            [
                _LocalTemporalBranch(hidden_size, kernel_values[idx], dilation_values[idx], dropout)
                for idx in range(3)
            ]
        )
        self.scale_logits = nn.Parameter(torch.zeros(3))
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, length, num_features, hidden_size = tokens.shape
        values = tokens.permute(0, 2, 3, 1).reshape(batch_size * num_features, hidden_size, length)
        branch_outputs = [branch(values) for branch in self.branches]
        weights = torch.softmax(self.scale_logits, dim=0)
        fused = sum(weights[idx] * branch_outputs[idx] for idx in range(3))

        def restore(sequence: torch.Tensor) -> torch.Tensor:
            return sequence.reshape(batch_size, num_features, hidden_size, length).permute(0, 3, 1, 2)

        fine, medium, coarse = (restore(branch_outputs[idx]) for idx in range(3))
        fused_tokens = self.output_norm(restore(fused))
        return fused_tokens, fine, medium, coarse


class StaticConditionalLayerNorm(nn.Module):
    """Static FiLM-style conditional LayerNorm for tokens [B, T, K, D]."""

    def __init__(self, hidden_size: int, context_size: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        context_size = context_size or hidden_size
        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Linear(context_size, hidden_size)
        self.beta = nn.Linear(context_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, tokens: torch.Tensor, static_context: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(static_context).view(static_context.shape[0], 1, 1, -1)
        beta = self.beta(static_context).view(static_context.shape[0], 1, 1, -1)
        conditioned = gamma * self.norm(tokens) + beta
        return tokens + self.residual_scale * self.dropout(conditioned)


class DecomposedTeacherPredictionHeads(nn.Module):
    """MRT heads for trend, residual, and direct sequence predictions."""

    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.trend_head = HierarchicalTrendDecoder(hidden_size, dropout)
        self.detail_head = ResidualSpectralDecoder(hidden_size, dropout)
        self.direct_head = nn.Sequential(
            GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout=dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, trend_features: torch.Tensor, detail_features: torch.Tensor, direct_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_trend = self.trend_head(trend_features)
        pred_detail, detail_spectrum = self.detail_head(detail_features)
        pred_direct = self.direct_head(direct_features).squeeze(-1)
        return pred_trend, pred_detail, pred_direct, detail_spectrum


DEFAULT_PHYSIOLOGICAL_GROUPS = {
    "workload_group": ("power", "power_rate"),
    "respiratory_group": ("rf", "rf_rate"),
    "cardiovascular_group": ("hr", "hrr", "hr_rate", "hrr_rate", "p_syst", "p_diast"),
    "oxygenation_group": ("spo2",),
}


class PhysiologicalLagAlignment(nn.Module):
    """Causal per-variable lag alignment for local features [B, T, K, D]."""

    def __init__(
        self,
        hidden_size: int,
        lags: Sequence[int] = (0, 1, 2, 3, 4, 5, 6),
        lag_hidden_dim: int | None = None,
        temperature: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        lag_values = [max(int(lag), 0) for lag in lags]
        if not lag_values:
            lag_values = [0]
        self.lags = tuple(lag_values)
        self.temperature = max(float(temperature), 1e-6)
        score_dim = int(lag_hidden_dim or hidden_size)
        self.query_projection = nn.Linear(hidden_size, score_dim, bias=False)
        self.lag_projection = nn.Linear(hidden_size, score_dim, bias=False)
        self.context_projection = nn.Linear(hidden_size, score_dim, bias=False)
        self.lag_embedding = nn.Embedding(len(self.lags), score_dim)
        self.score = nn.Linear(score_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def _shift_features(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, length, num_features, hidden_size = features.shape
        shifted = []
        valid = []
        for lag in self.lags:
            lagged = features.new_zeros((batch_size, length, num_features, hidden_size))
            lag_valid = torch.zeros((length,), dtype=torch.bool, device=features.device)
            if lag == 0:
                lagged = features
                lag_valid[:] = True
            elif lag < length:
                lagged[:, lag:] = features[:, : length - lag]
                lag_valid[lag:] = True
            shifted.append(lagged)
            valid.append(lag_valid)
        return torch.stack(shifted, dim=3), torch.stack(valid, dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        static_context: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shifted, valid_lags = self._shift_features(features)
        query = self.query_projection(features).unsqueeze(3)
        lag_repr = self.lag_projection(shifted)
        context = self.context_projection(static_context).view(static_context.shape[0], 1, 1, 1, -1)
        lag_ids = torch.arange(len(self.lags), device=features.device)
        lag_bias = self.lag_embedding(lag_ids).view(1, 1, 1, len(self.lags), -1)
        scores = self.score(torch.tanh(query + lag_repr + context + lag_bias)).squeeze(-1)
        scores = self.dropout(scores) / self.temperature

        valid_mask = valid_lags.view(1, features.shape[1], 1, len(self.lags)).expand_as(scores)
        if padding_mask is not None:
            if padding_mask.dim() == 2:
                current_valid = ~padding_mask.bool()
                valid_mask = valid_mask & current_valid.view(current_valid.shape[0], current_valid.shape[1], 1, 1)
                for lag_idx, lag in enumerate(self.lags):
                    source_valid = torch.zeros_like(current_valid)
                    if lag == 0:
                        source_valid = current_valid
                    elif lag < current_valid.shape[1]:
                        source_valid[:, lag:] = current_valid[:, : current_valid.shape[1] - lag]
                    valid_mask[..., lag_idx] = valid_mask[..., lag_idx] & source_valid.unsqueeze(-1)
            elif padding_mask.dim() == 3:
                current_valid = ~padding_mask.bool()
                valid_mask = valid_mask & current_valid.unsqueeze(-1)
                for lag_idx, lag in enumerate(self.lags):
                    source_valid = torch.zeros_like(current_valid)
                    if lag == 0:
                        source_valid = current_valid
                    elif lag < current_valid.shape[1]:
                        source_valid[:, lag:] = current_valid[:, : current_valid.shape[1] - lag]
                    valid_mask[..., lag_idx] = valid_mask[..., lag_idx] & source_valid

        masked_scores = scores.masked_fill(~valid_mask, -1.0e4)
        lag_weights = torch.softmax(masked_scores, dim=-1) * valid_mask.float()
        weight_sum = lag_weights.sum(dim=-1, keepdim=True)
        fallback = torch.softmax(scores, dim=-1)
        lag_weights = torch.where(weight_sum > 1e-8, lag_weights / weight_sum.clamp_min(1e-8), fallback)
        aligned = (lag_weights.unsqueeze(-1) * shifted).sum(dim=3)
        return aligned, lag_weights


class VTTargetQueryEncoder(nn.Module):
    """VT-centred cross-attention over dynamic variable tokens."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_positions: int = 512,
        use_group_tokens: bool = False,
        dynamic_feature_names: Sequence[str] | None = None,
        variable_groups: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        super().__init__()
        heads = num_heads if hidden_size % num_heads == 0 else 1
        self.target_embedding = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.target_embedding, mean=0.0, std=0.02)
        self.position_embedding = nn.Embedding(max_positions, hidden_size)
        self.static_projection = nn.Linear(hidden_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.ffn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout=dropout)
        self.max_positions = int(max_positions)
        self.use_group_tokens = bool(use_group_tokens)
        self.dynamic_feature_names = list(dynamic_feature_names or [])
        self.variable_groups = {
            name: tuple(values)
            for name, values in (variable_groups or DEFAULT_PHYSIOLOGICAL_GROUPS).items()
        }
        self.group_to_indices = self._build_group_indices()

    def _build_group_indices(self) -> Dict[str, List[int]]:
        name_to_idx = {name: idx for idx, name in enumerate(self.dynamic_feature_names)}
        groups: Dict[str, List[int]] = {}
        for group_name, feature_names in self.variable_groups.items():
            indices = [name_to_idx[name] for name in feature_names if name in name_to_idx]
            if indices:
                groups[group_name] = indices
        return groups

    def _append_group_tokens(self, features: torch.Tensor) -> torch.Tensor:
        if not self.use_group_tokens or not self.group_to_indices:
            return features
        group_tokens = [
            features[:, :, indices].mean(dim=2, keepdim=True)
            for indices in self.group_to_indices.values()
        ]
        return torch.cat([features] + group_tokens, dim=2)

    def forward(
        self,
        aligned_features: torch.Tensor,
        static_context: torch.Tensor,
        stage: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del stage
        batch_size, length, num_features, hidden_size = aligned_features.shape
        positions = torch.arange(length, device=aligned_features.device).clamp(max=self.max_positions - 1)
        query = (
            self.target_embedding.view(1, 1, hidden_size)
            + self.position_embedding(positions).view(1, length, hidden_size)
            + self.static_projection(static_context).unsqueeze(1)
        )
        keys = self._append_group_tokens(aligned_features)
        query_flat = query.reshape(batch_size * length, 1, hidden_size)
        key_flat = keys.reshape(batch_size * length, keys.shape[2], hidden_size)
        attended, attention = self.attention(
            query_flat,
            key_flat,
            key_flat,
            need_weights=True,
            average_attn_weights=False,
        )
        context = self.norm(query_flat + attended)
        context = self.ffn(context).reshape(batch_size, length, hidden_size)
        attention = attention.reshape(batch_size, attention.shape[1], length, keys.shape[2])
        variable_attention = attention[..., :num_features]
        variable_attention = variable_attention / variable_attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return context, variable_attention


class LearnableWaveletEncoder(nn.Module):
    """Lightweight learnable multi-scale wavelet-like encoder with time localization."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        kernel_sizes: Sequence[int] = (3, 7, 15),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim or input_dim)
        kernels = [max(int(kernel), 1) for kernel in kernel_sizes]
        if not kernels:
            kernels = [3, 7, 15]
        self.kernel_sizes = tuple(kernels)
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.scalar_projection = nn.Linear(1, self.hidden_dim)
        self.filters = nn.ParameterList(
            [nn.Parameter(torch.empty(self.hidden_dim, 1, kernel)) for kernel in self.kernel_sizes]
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for scale_idx, weight in enumerate(self.filters):
            kernel = weight.shape[-1]
            with torch.no_grad():
                weight.zero_()
                if scale_idx == 0:
                    weight.fill_(1.0 / float(kernel))
                elif scale_idx == 1:
                    center = kernel // 2
                    weight[:, 0, :center] = -1.0 / max(float(center), 1.0)
                    weight[:, 0, center + 1 :] = 1.0 / max(float(kernel - center - 1), 1.0)
                else:
                    center = kernel // 2
                    weight[:, 0, center] = 1.0
                    weight[:, 0, :] -= 1.0 / float(kernel)

    def _project(self, values: torch.Tensor) -> torch.Tensor:
        if values.dim() == 2:
            return self.scalar_projection(values.unsqueeze(-1).unsqueeze(2))
        if values.dim() == 3:
            return self.scalar_projection(values.unsqueeze(-1))
        if values.dim() == 4:
            if values.shape[-1] == self.input_dim:
                return self.input_projection(values)
            if values.shape[-1] == 1:
                return self.scalar_projection(values)
        raise ValueError(f"LearnableWaveletEncoder expects [B,T], [B,T,K], or [B,T,K,D], got {tuple(values.shape)}")

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        tokens = self._project(values)
        batch_size, length, num_features, hidden_dim = tokens.shape
        conv_input = tokens.permute(0, 2, 3, 1).reshape(batch_size * num_features, hidden_dim, length)
        scale_outputs = []
        for weight in self.filters:
            normalized = weight / weight.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-6).view(hidden_dim, 1, 1)
            kernel = normalized.shape[-1]
            left = (kernel - 1) // 2
            right = kernel - 1 - left
            padded = F.pad(conv_input, (left, right), mode="replicate")
            filtered = F.conv1d(padded, normalized, groups=hidden_dim)
            restored = filtered.reshape(batch_size, num_features, hidden_dim, length).permute(0, 3, 1, 2)
            scale_outputs.append(self.dropout(self.norm(restored)))
        return torch.stack(scale_outputs, dim=3)


class SpectralCoherenceAttention(nn.Module):
    """Target-centric spectral coherence over variables and wavelet scales."""

    def __init__(
        self,
        wavelet_dim: int,
        hidden_size: int,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.wavelet_projection = nn.Linear(wavelet_dim, hidden_size)
        self.target_projection = nn.Linear(hidden_size, hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.context_ffn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.eps = float(eps)

    def forward(
        self,
        wavelet_features: torch.Tensor,
        target_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = self.wavelet_projection(wavelet_features)
        target = self.target_projection(target_context).unsqueeze(2).unsqueeze(3)
        projected_norm = F.normalize(projected, dim=-1, eps=self.eps)
        target_norm = F.normalize(target, dim=-1, eps=self.eps)
        scores = (projected_norm * target_norm).sum(dim=-1)
        flat_attention = torch.softmax(scores.flatten(start_dim=2), dim=-1).view_as(scores)
        dropped_attention = self.dropout(flat_attention)
        dropped_sum = dropped_attention.sum(dim=(2, 3), keepdim=True)
        flat_attention = torch.where(
            dropped_sum > self.eps,
            dropped_attention / dropped_sum.clamp_min(self.eps),
            flat_attention,
        )
        spectral_context = (flat_attention.unsqueeze(-1) * projected).sum(dim=(2, 3))
        spectral_context = self.context_ffn(self.context_norm(spectral_context + target_context))
        scale_context = (flat_attention.unsqueeze(-1) * projected).sum(dim=2)
        scale_attention = flat_attention.sum(dim=2)
        variable_attention = flat_attention.sum(dim=3)
        return spectral_context, flat_attention, variable_attention, scale_attention, scale_context


class StaticSubjectPromptEncoder(nn.Module):
    """Create a lightweight subject prompt from static context."""

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.prompt = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, static_context: torch.Tensor) -> torch.Tensor:
        return self.norm(static_context + self.prompt(static_context))


class LocalPhysiologicalRelationBranch(nn.Module):
    """Grouped local variable relation branch over the [time, variable] field."""

    def __init__(
        self,
        hidden_size: int,
        dynamic_feature_names: Sequence[str],
        variable_groups: Mapping[str, Sequence[str]] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dynamic_feature_names = list(dynamic_feature_names)
        groups = variable_groups or DEFAULT_PHYSIOLOGICAL_GROUPS
        name_to_idx = {name: idx for idx, name in enumerate(self.dynamic_feature_names)}
        self.group_to_indices = {
            group_name: [name_to_idx[name] for name in names if name in name_to_idx]
            for group_name, names in groups.items()
        }
        self.group_to_indices = {name: indices for name, indices in self.group_to_indices.items() if indices}
        self.group_temporal = CausalConvBlock(hidden_size, kernel_size=3, dilation=1, dropout=dropout)
        self.variable_projection = nn.Linear(hidden_size, hidden_size)
        self.group_projection = nn.Linear(hidden_size, hidden_size)
        self.context_projection = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)
        self.fusion = GatedResidualNetwork(hidden_size * 2, hidden_size, hidden_size, dropout=dropout)

    def _group_context(self, features: torch.Tensor) -> torch.Tensor:
        if not self.group_to_indices:
            return features.mean(dim=2)
        group_tokens = [features[:, :, indices].mean(dim=2) for indices in self.group_to_indices.values()]
        group_stack = torch.stack(group_tokens, dim=2)
        return group_stack.mean(dim=2)

    def forward(self, features: torch.Tensor, target_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        group_sequence = self.group_temporal(self._group_context(features))
        scores = self.score(
            torch.tanh(
                self.variable_projection(features)
                + self.group_projection(group_sequence).unsqueeze(2)
                + self.context_projection(target_context).unsqueeze(2)
            )
        ).squeeze(-1)
        attention = torch.softmax(scores, dim=-1)
        variable_context = (attention.unsqueeze(-1) * features).sum(dim=2)
        return self.fusion(torch.cat([variable_context, group_sequence], dim=-1)), attention


class MRT(nn.Module):
    """Multi-scale Relational Teacher used by TidalDistill."""

    def __init__(
        self,
        feature_columns: Sequence[str],
        static_features: Sequence[str],
        dynamic_features: Sequence[str],
        static_categorical: Sequence[str],
        category_counts: Mapping[str, int],
        hidden_size: int = 32,
        embedding_dim: int = 8,
        num_attention_heads: int = 4,
        dropout: float = 0.2,
        multiscale_factors: Sequence[int] = (1, 2, 4, 8),
        use_static_context: bool = True,
        teacher_layers: int = 3,
        use_point_embedding: bool = True,
        use_feature_identity: bool = True,
        use_multiscale_local_encoder: bool = True,
        use_static_conditional_norm: bool = True,
        teacher_d_model: int | None = None,
        teacher_feature_id_dim: int = 8,
        teacher_local_kernels: Sequence[int] = (3, 5, 9),
        teacher_local_dilations: Sequence[int] = (1, 1, 2),
        teacher_local_dropout: float | None = None,
        teacher_decomp_alpha: float = 0.7,
        use_lag_alignment: bool = True,
        lag_steps: Sequence[int] = (0, 1, 2, 3, 4, 5, 6),
        lag_hidden_dim: int | None = None,
        lag_temperature: float = 1.0,
        relation_mode: str = "target_query",
        use_group_tokens: bool = False,
        target_query_heads: int | None = None,
        target_query_dropout: float | None = None,
        use_wavelet_encoder: bool = True,
        wavelet_num_scales: int = 3,
        wavelet_kernel_sizes: Sequence[int] = (3, 7, 15),
        wavelet_hidden_dim: int | None = None,
        use_spectral_coherence: bool = True,
        spectral_dropout: float | None = None,
        use_local_physiological_relation: bool = True,
        teacher_prediction_mode: str = "decomposed",
        **_: object,
    ) -> None:
        super().__init__()
        hidden_size = int(teacher_d_model or hidden_size)
        self.feature_columns = list(feature_columns)
        self.static_features = list(static_features)
        self.dynamic_features = list(dynamic_features)
        self.feature_to_idx = {name: idx for idx, name in enumerate(self.feature_columns)}
        self.dynamic_indices = [self.feature_to_idx[name] for name in self.dynamic_features]
        self.use_static_context = bool(use_static_context)
        self.multiscale_factors = [max(int(scale), 1) for scale in multiscale_factors]
        self.hidden_size = int(hidden_size)
        self.max_positions = 512
        self.use_point_embedding = bool(use_point_embedding)
        self.use_multiscale_local_encoder = bool(use_multiscale_local_encoder)
        self.use_static_conditional_norm = bool(use_static_conditional_norm)
        self.teacher_decomp_alpha = float(max(0.0, min(1.0, teacher_decomp_alpha)))
        self.use_lag_alignment = bool(use_lag_alignment)
        self.relation_mode = str(relation_mode).lower()
        if self.relation_mode not in {"itransformer", "target_query", "hybrid"}:
            self.relation_mode = "target_query"
        self.use_wavelet_encoder = bool(use_wavelet_encoder)
        self.use_spectral_coherence = bool(use_spectral_coherence)
        self.use_local_physiological_relation = bool(use_local_physiological_relation)
        self.teacher_prediction_mode = str(teacher_prediction_mode).lower()
        if self.teacher_prediction_mode not in {"decomposed", "direct"}:
            raise ValueError(f"Unknown teacher prediction mode: {teacher_prediction_mode}")
        wavelet_kernels = tuple(int(kernel) for kernel in wavelet_kernel_sizes)
        if wavelet_num_scales > 0:
            wavelet_kernels = wavelet_kernels[: int(wavelet_num_scales)]
        if not wavelet_kernels:
            wavelet_kernels = (3, 7, 15)
        self.wavelet_num_scales = len(wavelet_kernels)
        self.wavelet_hidden_dim = int(wavelet_hidden_dim or hidden_size)

        static_continuous = [name for name in self.static_features if name not in set(static_categorical)]
        self.static_encoder = (
            StaticConditionalEncoder(
                feature_to_idx=self.feature_to_idx,
                static_categorical=static_categorical,
                static_continuous=static_continuous,
                category_counts=category_counts,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                dropout=dropout,
            )
            if self.use_static_context
            else None
        )
        self.dynamic_value_projection = nn.Linear(1, hidden_size)
        self.dynamic_step_projection = nn.Linear(len(self.dynamic_features), hidden_size)
        self.variable_embedding = nn.Embedding(len(self.dynamic_features), hidden_size)
        self.position_embedding = nn.Embedding(self.max_positions, hidden_size)
        self.variable_blocks = nn.ModuleList(
            [VariableTransformerBlock(hidden_size, num_attention_heads, dropout) for _ in range(max(int(teacher_layers), 1))]
        )
        self.temporal_refiner = TemporalSelfAttentionBlock(hidden_size, num_attention_heads, dropout)
        self.context_fusion = GatedResidualNetwork(hidden_size * 3, hidden_size, hidden_size, dropout=dropout)
        self.scale_score = nn.Linear(hidden_size, 1)
        self.point_embedding = PhysiologicalPointEmbedding(
            num_features=len(self.dynamic_features),
            hidden_size=hidden_size,
            feature_id_dim=teacher_feature_id_dim,
            dropout=dropout,
            use_feature_identity=use_feature_identity,
            max_positions=self.max_positions,
            num_stages=4,
        )
        self.subject_prompt_encoder = StaticSubjectPromptEncoder(hidden_size, dropout=dropout)
        self.static_conditional_norm = StaticConditionalLayerNorm(hidden_size, hidden_size, dropout=dropout)
        self.local_temporal_encoder = MultiScaleLocalTemporalEncoder(
            hidden_size=hidden_size,
            kernels=teacher_local_kernels,
            dilations=teacher_local_dilations,
            dropout=dropout if teacher_local_dropout is None else float(teacher_local_dropout),
        )
        self.lag_alignment = PhysiologicalLagAlignment(
            hidden_size=hidden_size,
            lags=lag_steps,
            lag_hidden_dim=lag_hidden_dim,
            temperature=lag_temperature,
            dropout=dropout,
        )
        self.target_query_encoder = VTTargetQueryEncoder(
            hidden_size=hidden_size,
            num_heads=num_attention_heads if target_query_heads is None else int(target_query_heads),
            dropout=dropout if target_query_dropout is None else float(target_query_dropout),
            max_positions=self.max_positions,
            use_group_tokens=use_group_tokens,
            dynamic_feature_names=self.dynamic_features,
        )
        self.target_relation_fusion = GatedResidualNetwork(hidden_size * 2, hidden_size, hidden_size, dropout=dropout)
        self.wavelet_encoder = LearnableWaveletEncoder(
            input_dim=hidden_size,
            hidden_dim=self.wavelet_hidden_dim,
            kernel_sizes=wavelet_kernels,
            dropout=dropout,
        )
        self.spectral_coherence = SpectralCoherenceAttention(
            wavelet_dim=self.wavelet_hidden_dim,
            hidden_size=hidden_size,
            dropout=dropout if spectral_dropout is None else float(spectral_dropout),
        )
        self.local_relation_branch = LocalPhysiologicalRelationBranch(
            hidden_size=hidden_size,
            dynamic_feature_names=self.dynamic_features,
            dropout=dropout,
        )
        self.local_relation_fusion = GatedResidualNetwork(hidden_size * 2, hidden_size, hidden_size, dropout=dropout)
        self.multilevel_fusion = GatedResidualNetwork(hidden_size * 5, hidden_size, hidden_size, dropout=dropout)
        self.delta_step_projection = nn.Linear(len(self.dynamic_features), hidden_size)
        self.trend_feature_fusion = GatedResidualNetwork(hidden_size * 4, hidden_size, hidden_size, dropout=dropout)
        self.detail_feature_fusion = GatedResidualNetwork(hidden_size * 5, hidden_size, hidden_size, dropout=dropout)
        self.direct_feature_fusion = GatedResidualNetwork(hidden_size * 3, hidden_size, hidden_size, dropout=dropout)
        self.decomposed_heads = DecomposedTeacherPredictionHeads(hidden_size, dropout)
        self.last_outputs: Dict[str, object] = {}

    @staticmethod
    def _normalize_positive(scores: torch.Tensor, dim: int) -> torch.Tensor:
        scores = scores.clamp_min(0.0)
        denom = scores.sum(dim=dim, keepdim=True)
        uniform = torch.full_like(scores, 1.0 / max(scores.shape[dim], 1))
        return torch.where(denom > 1e-8, scores / (denom + 1e-8), uniform)

    def _context(self, x: torch.Tensor) -> torch.Tensor:
        if self.static_encoder is None:
            return x.new_zeros((x.shape[0], self.hidden_size))
        return self.static_encoder(x)

    def _infer_stage(self, dynamic_raw: torch.Tensor) -> torch.Tensor:
        if "power" not in self.dynamic_features:
            return torch.zeros(dynamic_raw.shape[:2], dtype=torch.long, device=dynamic_raw.device)
        power_idx = self.dynamic_features.index("power")
        power = dynamic_raw[:, :, power_idx]
        shifted_power = power - power.min(dim=1, keepdim=True).values
        max_power = shifted_power.max(dim=1, keepdim=True).values.clamp_min(1e-6)
        relative_power = shifted_power / max_power
        stage = torch.zeros_like(power, dtype=torch.long)
        stage = torch.where(relative_power >= 0.33, torch.ones_like(stage), stage)
        stage = torch.where(relative_power >= 0.66, torch.full_like(stage, 2), stage)
        if "power_rate" in self.dynamic_features:
            power_rate = dynamic_raw[:, :, self.dynamic_features.index("power_rate")]
            max_power_idx = shifted_power.argmax(dim=1)
            time_ids = torch.arange(dynamic_raw.shape[1], device=dynamic_raw.device).view(1, -1)
            recovery = (time_ids > max_power_idx.view(-1, 1)) & (power_rate < 0)
            stage = torch.where(recovery, torch.full_like(stage, 3), stage)
        return stage

    def _multi_scale_sequences(self, sequence: torch.Tensor) -> List[torch.Tensor]:
        values = sequence.unsqueeze(1)
        outputs: List[torch.Tensor] = []
        for scale in self.multiscale_factors:
            if scale == 1:
                outputs.append(sequence)
            else:
                outputs.append(F.avg_pool1d(values, kernel_size=scale, stride=scale, ceil_mode=True).squeeze(1))
        return outputs

    def _scale_importance(self, sequence: torch.Tensor) -> torch.Tensor:
        pooled: List[torch.Tensor] = []
        values = sequence.transpose(1, 2)
        for scale in self.multiscale_factors:
            if scale == 1:
                pooled.append(sequence.mean(dim=1))
            else:
                coarse = F.avg_pool1d(values, kernel_size=scale, stride=scale, ceil_mode=True).transpose(1, 2)
                pooled.append(coarse.mean(dim=1))
        scale_repr = torch.stack(pooled, dim=1)
        return torch.softmax(self.scale_score(scale_repr).squeeze(-1), dim=-1)

    @staticmethod
    def _causal_smooth_feature(sequence: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
        if sequence.shape[1] <= 1:
            return sequence
        kernel = max(int(kernel_size), 1)
        values = sequence.transpose(1, 2)
        padded = F.pad(values, (kernel - 1, 0), mode="replicate")
        return F.avg_pool1d(padded, kernel_size=kernel, stride=1).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor | List[torch.Tensor] | Dict[str, object]]:
        context = self._context(x)
        subject_prompt = self.subject_prompt_encoder(context)
        dynamic_raw = x[:, :, self.dynamic_indices]
        batch_size, target_length, dynamic_size = dynamic_raw.shape
        stage = self._infer_stage(dynamic_raw)
        observation_mask = torch.ones_like(dynamic_raw)
        time_since_last_observation = torch.zeros_like(dynamic_raw)

        if self.use_point_embedding:
            point_tokens = self.point_embedding(
                dynamic_raw,
                stage=stage,
                observation_mask=observation_mask,
                time_since_last_observation=time_since_last_observation,
            )
        else:
            positions = torch.arange(target_length, device=x.device).clamp(max=self.max_positions - 1)
            time_pos = self.position_embedding(positions).view(1, target_length, 1, -1)
            variable_ids = torch.arange(dynamic_size, device=x.device)
            variable_pos = self.variable_embedding(variable_ids).view(1, 1, dynamic_size, -1)
            point_tokens = self.dynamic_value_projection(dynamic_raw.unsqueeze(-1)) + time_pos + variable_pos

        if self.use_static_conditional_norm and self.static_encoder is not None:
            point_tokens = self.static_conditional_norm(point_tokens, subject_prompt)

        if self.use_multiscale_local_encoder:
            local_tokens, fine_tokens, medium_tokens, coarse_tokens = self.local_temporal_encoder(point_tokens)
        else:
            local_tokens = fine_tokens = medium_tokens = coarse_tokens = point_tokens

        if self.use_static_conditional_norm and self.static_encoder is not None:
            local_tokens = self.static_conditional_norm(local_tokens, subject_prompt)
            fine_tokens = self.static_conditional_norm(fine_tokens, subject_prompt)
            medium_tokens = self.static_conditional_norm(medium_tokens, subject_prompt)
            coarse_tokens = self.static_conditional_norm(coarse_tokens, subject_prompt)

        if self.use_lag_alignment:
            aligned_tokens, lag_weights = self.lag_alignment(local_tokens, subject_prompt, padding_mask=None)
        else:
            aligned_tokens = local_tokens
            lag_weights = local_tokens.new_ones((*local_tokens.shape[:3], 1))

        if self.use_static_conditional_norm and self.static_encoder is not None:
            aligned_tokens = self.static_conditional_norm(aligned_tokens, subject_prompt)

        variable_tokens = aligned_tokens.mean(dim=1) + subject_prompt.unsqueeze(1)
        relation_map = None
        for block in self.variable_blocks:
            variable_tokens, relation_map = block(variable_tokens)

        local_sequence = local_tokens.mean(dim=2)
        aligned_sequence = aligned_tokens.mean(dim=2)
        fine_sequence = fine_tokens.mean(dim=2)
        medium_sequence = medium_tokens.mean(dim=2)
        coarse_sequence = coarse_tokens.mean(dim=2)
        lowfreq_sequence = self._causal_smooth_feature(coarse_sequence, kernel_size=9)
        itransformer_context = variable_tokens.mean(dim=1).unsqueeze(1).expand(-1, target_length, -1)
        if self.relation_mode == "itransformer":
            target_query_context = x.new_zeros((batch_size, target_length, self.hidden_size))
            target_variable_attention = x.new_zeros((batch_size, 1, target_length, dynamic_size))
        else:
            target_query_context, target_variable_attention = self.target_query_encoder(aligned_tokens, subject_prompt)
        if self.relation_mode == "itransformer":
            relation_context = itransformer_context
            variable_attention = (
                relation_map
                if relation_map is not None
                else x.new_zeros((batch_size, 1, dynamic_size, dynamic_size))
            )
        elif self.relation_mode == "hybrid":
            relation_context = self.target_relation_fusion(torch.cat([itransformer_context, target_query_context], dim=-1))
            variable_attention = target_variable_attention
        else:
            relation_context = target_query_context
            variable_attention = target_variable_attention
        if self.use_local_physiological_relation:
            local_relation_context, local_relation_attention = self.local_relation_branch(aligned_tokens, relation_context)
            relation_context = self.local_relation_fusion(torch.cat([relation_context, local_relation_context], dim=-1))
        else:
            local_relation_context = x.new_zeros((batch_size, target_length, self.hidden_size))
            local_relation_attention = x.new_zeros((batch_size, target_length, dynamic_size))
        context_sequence = context.unsqueeze(1).expand(-1, target_length, -1)
        zero_context = x.new_zeros((batch_size, target_length, self.hidden_size))
        if self.use_wavelet_encoder:
            wavelet_features = self.wavelet_encoder(aligned_tokens)
        else:
            wavelet_features = x.new_zeros(
                (batch_size, target_length, dynamic_size, self.wavelet_num_scales, self.wavelet_hidden_dim)
            )
        if self.use_wavelet_encoder and self.use_spectral_coherence:
            (
                spectral_context,
                spectral_attention,
                spectral_variable_attention,
                spectral_scale_attention,
                spectral_scale_context,
            ) = self.spectral_coherence(wavelet_features, relation_context)
        else:
            spectral_context = zero_context
            spectral_attention = x.new_zeros((batch_size, target_length, dynamic_size, self.wavelet_num_scales))
            spectral_variable_attention = x.new_zeros((batch_size, target_length, dynamic_size))
            spectral_scale_attention = x.new_zeros((batch_size, target_length, self.wavelet_num_scales))
            spectral_scale_context = x.new_zeros((batch_size, target_length, self.wavelet_num_scales, self.hidden_size))

        low_spectral_context = spectral_scale_context[:, :, 0] if self.wavelet_num_scales >= 1 else zero_context
        if self.wavelet_num_scales >= 3:
            detail_spectral_context = spectral_scale_context[:, :, 1:].mean(dim=2)
        elif self.wavelet_num_scales == 2:
            detail_spectral_context = spectral_scale_context[:, :, 1]
        elif self.wavelet_num_scales == 1:
            detail_spectral_context = spectral_scale_context[:, :, 0]
        else:
            detail_spectral_context = zero_context

        step_tokens = self.dynamic_step_projection(dynamic_raw) + local_sequence
        fused_sequence = self.context_fusion(torch.cat([step_tokens, relation_context, context_sequence], dim=-1))
        fused_sequence, temporal_attention = self.temporal_refiner(fused_sequence)
        fused_sequence = self.multilevel_fusion(
            torch.cat([local_sequence, aligned_sequence, relation_context, spectral_context, fused_sequence], dim=-1)
        )

        delta_raw = torch.zeros_like(dynamic_raw)
        if target_length > 1:
            delta_raw[:, 1:] = dynamic_raw[:, 1:] - dynamic_raw[:, :-1]
        delta_sequence = self.delta_step_projection(delta_raw)

        trend_features = self.trend_feature_fusion(
            torch.cat([coarse_sequence, low_spectral_context, relation_context + lowfreq_sequence, context_sequence], dim=-1)
        )
        detail_features = self.detail_feature_fusion(
            torch.cat(
                [fine_sequence, medium_sequence, aligned_sequence, detail_spectral_context, relation_context + delta_sequence],
                dim=-1,
            )
        )
        direct_features = self.direct_feature_fusion(torch.cat([relation_context, fused_sequence, spectral_context], dim=-1))
        pred_trend, pred_detail_raw, pred_direct, residual_spectrum = self.decomposed_heads(
            trend_features,
            detail_features,
            direct_features,
        )

        if self.teacher_prediction_mode == "direct":
            pred_trend = torch.zeros_like(pred_direct)
            pred_detail_raw = torch.zeros_like(pred_direct)
            pred_detail = pred_detail_raw
            residual_spectrum = torch.abs(torch.fft.rfft(pred_detail, dim=1))
            pred_seq = pred_direct
            residual_gate = torch.zeros_like(pred_seq)
        else:
            pred_detail = pred_detail_raw
            pred_decomp = pred_trend + pred_detail
            pred_seq = self.teacher_decomp_alpha * pred_decomp + (1.0 - self.teacher_decomp_alpha) * pred_direct
            residual_gate = torch.full_like(pred_seq, self.teacher_decomp_alpha)

        scale_importance = self._scale_importance(fused_sequence)
        multiscale_sequences = self._multi_scale_sequences(pred_seq)
        outputs: Dict[str, torch.Tensor | List[torch.Tensor] | Dict[str, object]] = {
            "pred": pred_seq[:, -1],
            "pred_last": pred_seq[:, -1],
            "pred_seq": pred_seq,
            "pred_trend": pred_trend,
            "pred_detail": pred_detail,
            "pred_direct": pred_direct,
            "trend_seq": pred_trend,
            "residual_seq": pred_detail,
            "detail_seq": pred_detail,
            "raw_residual_seq": pred_detail_raw,
            "raw_detail_seq": pred_detail_raw,
            "residual_gate": residual_gate,
            "detail_gate": residual_gate,
            "residual_spectrum": residual_spectrum,
            "trend_repr": trend_features.mean(dim=1),
            "residual_repr": detail_features.mean(dim=1),
            "variable_repr": variable_tokens,
            "relation_map": relation_map if relation_map is not None else x.new_zeros((batch_size, 1, dynamic_size, dynamic_size)),
            "relation_fused": variable_tokens.mean(dim=1),
            "h_fused": fused_sequence,
            "h_teacher": fused_sequence,
            "seq_soft": pred_seq,
            "multiscale_sequences": multiscale_sequences,
            "scale_importance": scale_importance,
            "temporal_attention": temporal_attention,
            "fine_feature": fine_tokens,
            "medium_feature": medium_tokens,
            "coarse_feature": coarse_tokens,
            "lag_weights": lag_weights,
            "variable_attention": variable_attention,
            "target_variable_attention": target_variable_attention,
            "target_context": relation_context,
            "subject_prompt": subject_prompt,
            "stage": stage,
            "local_relation_attention": local_relation_attention,
            "local_relation_context": local_relation_context,
            "wavelet_features": wavelet_features,
            "spectral_context": spectral_context,
            "spectral_attention": spectral_attention,
            "spectral_variable_attention": spectral_variable_attention,
            "spectral_scale_attention": spectral_scale_attention,
            "auxiliary_outputs": {
                "local_tokens": local_tokens,
                "aligned_features": aligned_tokens,
                "lag_weights": lag_weights,
                "variable_attention": variable_attention,
                "target_variable_attention": target_variable_attention,
                "target_context": relation_context,
                "subject_prompt": subject_prompt,
                "stage": stage,
                "local_relation_attention": local_relation_attention,
                "local_relation_context": local_relation_context,
                "wavelet_features": wavelet_features,
                "spectral_context": spectral_context,
                "spectral_attention": spectral_attention,
                "spectral_variable_attention": spectral_variable_attention,
                "spectral_scale_attention": spectral_scale_attention,
                "fine_feature": fine_tokens,
                "medium_feature": medium_tokens,
                "coarse_feature": coarse_tokens,
                "delta_sequence": delta_sequence,
                "mode": "stage3" if self.use_wavelet_encoder else "stage2",
                "relation_mode": self.relation_mode,
            },
        }
        self.last_outputs = outputs
        return outputs


class DAE(nn.Module):
    """Dual-branch Adaptive Estimator used for TidalDistill inference."""

    def __init__(
        self,
        feature_columns: Sequence[str] = FEATURE_COLUMNS,
        static_features: Sequence[str] = STATIC_FEATURES,
        dynamic_features: Sequence[str] = DYNAMIC_FEATURES,
        static_categorical: Sequence[str] = STATIC_CATEGORICAL,
        category_counts: Mapping[str, int] = CATEGORY_COUNTS,
        hidden_size: int = 32,
        embedding_dim: int = 8,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_columns = list(feature_columns)
        self.static_features = list(static_features)
        self.dynamic_features = list(dynamic_features)
        self.feature_to_idx = {name: idx for idx, name in enumerate(self.feature_columns)}
        self.dynamic_indices = [self.feature_to_idx[name] for name in self.dynamic_features]
        self.hidden_size = int(hidden_size)

        static_continuous = [name for name in self.static_features if name not in set(static_categorical)]
        self.static_encoder = StaticConditionalEncoder(
            feature_to_idx=self.feature_to_idx,
            static_categorical=static_categorical,
            static_continuous=static_continuous,
            category_counts=category_counts,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.dynamic_value_projection = nn.Linear(1, hidden_size)
        self.feature_embedding = nn.Embedding(len(self.dynamic_features), hidden_size)
        self.dual_selector = DualFeatureSelection(hidden_size, dropout, mode="dual")
        self.trend_extractor = TFLSTMExtractor(hidden_size, num_attention_heads, dropout)
        self.detail_extractor = TFLSTMExtractor(hidden_size, num_attention_heads, dropout)
        self.variable_relation = VariableRelationStream(hidden_size, num_attention_heads, dropout)
        self.lightweight_fusion = GatedResidualNetwork(
            hidden_size * 3,
            hidden_size,
            hidden_size,
            dropout=dropout,
        )
        self.heads = TimeFusionPredictionHeads(hidden_size, dropout)
        self.last_outputs: Dict[str, object] = {}

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        if x.ndim != 3 or x.shape[1:] != (WINDOW_SIZE, len(self.feature_columns)):
            raise ValueError(
                f"Expected input shape [batch, {WINDOW_SIZE}, {len(self.feature_columns)}], "
                f"received {tuple(x.shape)}."
            )

        context = self.static_encoder(x)
        dynamic_raw = x[:, :, self.dynamic_indices]
        dynamic_embeddings = self.dynamic_value_projection(dynamic_raw.unsqueeze(-1))
        feature_ids = torch.arange(len(self.dynamic_features), device=x.device)
        dynamic_embeddings = dynamic_embeddings + self.feature_embedding(feature_ids).view(
            1, 1, -1, self.hidden_size
        )

        trend_input, detail_input, trend_weights, detail_weights = self.dual_selector(
            dynamic_embeddings,
            context,
        )
        trend_features, trend_attention = self.trend_extractor(trend_input)
        detail_features, detail_attention = self.detail_extractor(detail_input)
        relation_context, variable_tokens, relation_map = self.variable_relation(
            dynamic_embeddings,
            context,
        )
        context_sequence = context.unsqueeze(1).expand(-1, x.shape[1], -1)
        fused_sequence = self.lightweight_fusion(
            torch.cat(
                [trend_features, detail_features + relation_context, context_sequence],
                dim=-1,
            )
        )

        trend_seq, raw_residual_seq, residual_spectrum, pred_seq, residual_gate = self.heads(
            fused_sequence,
            context,
        )
        residual_seq = residual_gate * raw_residual_seq
        outputs: Dict[str, object] = {
            "pred": pred_seq[:, -1],
            "pred_seq": pred_seq,
            "trend_seq": trend_seq,
            "residual_seq": residual_seq,
            "raw_residual_seq": raw_residual_seq,
            "residual_gate": residual_gate,
            "residual_spectrum": residual_spectrum,
            "feature_gates_trend": trend_weights,
            "feature_gates_residual": detail_weights,
            "trend_attention": trend_attention,
            "detail_attention": detail_attention,
            "variable_attention": relation_map,
            "variable_features": variable_tokens,
            "fused_features": fused_sequence,
            "context": context,
        }
        self.last_outputs = outputs
        return outputs

    @torch.no_grad()
    def get_interpretability(self) -> Dict[str, object]:
        result: Dict[str, object] = {"dynamic_features": self.dynamic_features}
        for key, value in self.last_outputs.items():
            if torch.is_tensor(value):
                result[key] = value.detach().cpu()
        return result


def build_dae() -> DAE:
    """Construct the manuscript DAE inference architecture."""
    return DAE()


def build_mrt() -> MRT:
    """Construct the manuscript MRT architecture."""
    return MRT(
        feature_columns=FEATURE_COLUMNS,
        static_features=STATIC_FEATURES,
        dynamic_features=DYNAMIC_FEATURES,
        static_categorical=STATIC_CATEGORICAL,
        category_counts=CATEGORY_COUNTS,
        hidden_size=32,
        embedding_dim=8,
        num_attention_heads=4,
        dropout=0.1,
        multiscale_factors=(1, 2, 4, 8),
        teacher_local_kernels=(3, 5, 9),
        teacher_local_dilations=(1, 1, 2),
        teacher_decomp_alpha=0.7,
        lag_steps=(0, 1, 2, 3, 4, 5, 6),
        target_query_heads=4,
        wavelet_num_scales=3,
        wavelet_kernel_sizes=(3, 7, 15),
    )


def _load_state_dict(checkpoint: str | Path, map_location: str | torch.device) -> Dict[str, torch.Tensor]:
    try:
        state = torch.load(Path(checkpoint), map_location=map_location, weights_only=True)
    except TypeError:
        state = torch.load(Path(checkpoint), map_location=map_location)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must contain a PyTorch state dictionary.")
    return state


def load_dae(
    checkpoint: str | Path,
    map_location: str | torch.device = "cpu",
) -> DAE:
    """Load a DAE state dictionary for inference."""
    model = build_dae()
    model.load_state_dict(_load_state_dict(checkpoint, map_location), strict=True)
    model.eval()
    return model


def load_mrt(
    checkpoint: str | Path,
    map_location: str | torch.device = "cpu",
) -> MRT:
    """Load an MRT state dictionary for reference inference."""
    model = build_mrt()
    model.load_state_dict(_load_state_dict(checkpoint, map_location), strict=True)
    model.eval()
    return model
