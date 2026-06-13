"""Model definitions for triplet learning on precomputed CLIP frame embeddings.

TripletNet1 is a hierarchical video-to-artist encoder for the current data setup:
    artist
      -> up to 10 videos, zero-padded when fewer videos are available
      -> each video has 30 uniformly sampled CLIP frame embeddings

TripletNet1 uses a simple hierarchical Encoder-only Transformer:
    1. project CLIP frame embeddings to model_dim;
    2. encode frames inside each video with a TransformerEncoder;
    3. pool encoded frames into one video token;
    4. encode up to 10 video tokens with a masked TransformerEncoder;
    5. masked-pool video tokens, then projection head + BNNeck + L2 normalize.

TripletNet1 accepts tensors shaped either:
    (batch, videos, frames, d_model), recommended, or
    (batch, frames, d_model), legacy one-video/averaged format.

TripletNet2-4 keep the legacy 2D artist tensor path shaped
``(batch, seq_len, d_model)``.
"""

from __future__ import annotations

from typing import Dict, Type

import torch
import torch.nn.functional as F
from torch import Tensor, nn


DEFAULT_EMBEDDING_DIM = 768
DEFAULT_OUTPUT_DIM = 256
DEFAULT_MAX_SEQ_LEN = 30
DEFAULT_MAX_VIDEOS = 10


def _init_linear(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _select_nhead(d_model: int, preferred: int = 8) -> int:
    if d_model <= 0:
        raise ValueError("d_model must be positive")
    for candidate in (preferred, 8, 4, 2, 1):
        if candidate <= d_model and d_model % candidate == 0:
            return candidate
    return 1


def _make_group_norm(num_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=1, num_channels=num_channels)


def _infer_fc_in_features(forward_fn, seq_len: int, d_model: int) -> int:
    if seq_len <= 0 or d_model <= 0:
        raise ValueError("seq_len and d_model must be positive")
    with torch.no_grad():
        dummy = torch.zeros(2, seq_len, d_model)
        out = forward_fn(dummy)
        return out.flatten(1).shape[1]


def _make_transformer_encoder(encoder_layer: nn.TransformerEncoderLayer, num_layers: int) -> nn.TransformerEncoder:
    """Create a TransformerEncoder while disabling nested tensors when supported.

    Key-padding masks over padded videos are small here and disabling nested
    tensors avoids prototype API warnings on recent PyTorch builds.
    """
    try:
        return nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
    except TypeError:  # pragma: no cover - for older PyTorch versions
        return nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)


class SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that safely handles one-sample training mini-batches."""

    def forward(self, input: Tensor) -> Tensor:  # noqa: D401 - same semantics as BatchNorm1d
        if self.training and input.ndim >= 2 and int(input.size(0)) <= 1:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=self.momentum,
                eps=self.eps,
            )
        return super().forward(input)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model <= 0 or max_len <= 0:
            raise ValueError("d_model and max_len must be positive")
        self.position_embedding = nn.Parameter(torch.empty(1, max_len, d_model))
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, seq_len, dim), got {tuple(x.shape)}")
        seq_len = x.size(1)
        if seq_len > self.position_embedding.size(1):
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_len {self.position_embedding.size(1)}"
            )
        if x.size(2) != self.position_embedding.size(2):
            raise ValueError(
                f"Input feature dim {x.size(2)} does not match d_model {self.position_embedding.size(2)}"
            )
        return self.dropout(x + self.position_embedding[:, :seq_len, :])


class AttentiveStatsPool(nn.Module):
    """Attention-weighted mean + std pooling."""

    def __init__(self, d_model: int, hidden_dim: int | None = None, eps: float = 1e-6) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        hidden_dim = hidden_dim or min(256, max(64, d_model // 2))
        self.eps = eps
        self.attention = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                _init_linear(module)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, seq_len, dim), got {tuple(x.shape)}")
        weights = torch.softmax(self.attention(x), dim=1)
        mean = torch.sum(weights * x, dim=1)
        var = torch.sum(weights * (x - mean.unsqueeze(1)).pow(2), dim=1)
        std = torch.sqrt(var.clamp_min(self.eps))
        return torch.cat([mean, std], dim=1)


class MultiHeadAttentiveStatsPool(nn.Module):
    """Multi-view attention-weighted mean + std pooling."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        hidden_dim: int | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        hidden_dim = hidden_dim or min(256, max(64, d_model // 2))
        self.num_heads = num_heads
        self.eps = eps
        self.attention = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                _init_linear(module)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, seq_len, dim), got {tuple(x.shape)}")
        weights = torch.softmax(self.attention(x), dim=1).transpose(1, 2)
        mean = torch.bmm(weights, x)
        var = torch.sum(weights.unsqueeze(-1) * (x.unsqueeze(1) - mean.unsqueeze(2)).pow(2), dim=2)
        std = torch.sqrt(var.clamp_min(self.eps))
        return torch.cat([mean, std], dim=2).flatten(1)


class MaskedMultiHeadAttentiveStatsPool(nn.Module):
    """Multi-head attentive mean/std pooling over a padded set."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        hidden_dim: int | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        hidden_dim = hidden_dim or min(256, max(64, d_model // 2))
        self.num_heads = num_heads
        self.eps = eps
        self.attention = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                _init_linear(module)

    @staticmethod
    def _validate_mask(mask: Tensor, x: Tensor) -> Tensor:
        if mask.ndim != 2:
            raise ValueError(f"Expected mask shape (batch, seq_len), got {tuple(mask.shape)}")
        if mask.shape != x.shape[:2]:
            raise ValueError(f"Mask shape {tuple(mask.shape)} does not match input prefix {tuple(x.shape[:2])}")
        mask = mask.bool()
        if not mask.any(dim=1).all():
            mask = mask.clone()
            mask[~mask.any(dim=1), 0] = True
        return mask

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, seq_len, dim), got {tuple(x.shape)}")
        mask = self._validate_mask(mask, x)
        logits = self.attention(x)
        logits = logits.masked_fill(~mask.unsqueeze(-1), torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1).transpose(1, 2)
        weights = weights * mask.unsqueeze(1).to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(self.eps)
        mean = torch.bmm(weights, x)
        var = torch.sum(weights.unsqueeze(-1) * (x.unsqueeze(1) - mean.unsqueeze(2)).pow(2), dim=2)
        std = torch.sqrt(var.clamp_min(self.eps))
        return torch.cat([mean, std], dim=2).flatten(1)


class BaseTripletNet(nn.Module):
    def encode(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def forward_once(self, x: Tensor) -> Tensor:
        return self.encode(x)

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.encode(anchor), self.encode(positive), self.encode(negative)


class HierarchicalVideoArtistTripletNet(BaseTripletNet):
    """Hierarchical encoder for ``artist -> videos -> CLIP frame embeddings``."""

    def __init__(
        self,
        d_model: int = DEFAULT_EMBEDDING_DIM,
        output_dim: int = DEFAULT_OUTPUT_DIM,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        max_videos: int = DEFAULT_MAX_VIDEOS,
        model_dim: int = 384,
        nhead: int | None = None,
        frame_transformer_layers: int = 1,
        artist_transformer_layers: int = 1,
        dropout: float = 0.15,
        dim_feedforward: int | None = None,
        projection_hidden_dim: int = 768,
        projection_mid_dim: int = 512,
        frame_attention_heads: int = 4,
        artist_attention_heads: int = 4,
        video_dropout_p: float = 0.15,
        mask_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if d_model <= 0 or output_dim <= 0 or max_seq_len <= 0 or max_videos <= 0 or model_dim <= 0:
            raise ValueError("d_model, output_dim, max_seq_len, max_videos and model_dim must be positive")
        if frame_transformer_layers <= 0 or artist_transformer_layers <= 0:
            raise ValueError("transformer layer counts must be positive")
        if projection_hidden_dim <= 0 or projection_mid_dim <= 0:
            raise ValueError("projection hidden dimensions must be positive")
        if frame_attention_heads <= 0 or artist_attention_heads <= 0:
            raise ValueError("attention head counts must be positive")
        if not 0.0 <= video_dropout_p < 1.0:
            raise ValueError("video_dropout_p must be in [0, 1)")
        if mask_eps < 0:
            raise ValueError("mask_eps must be non-negative")

        nhead = _select_nhead(model_dim, preferred=8) if nhead is None else nhead
        if nhead <= 0 or model_dim % nhead != 0:
            raise ValueError(f"nhead={nhead} must be positive and divide model_dim={model_dim}")
        if dim_feedforward is None:
            dim_feedforward = 4 * model_dim

        self.d_model = d_model
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len
        self.max_videos = max_videos
        self.video_dropout_p = float(video_dropout_p)
        self.mask_eps = float(mask_eps)
        self.video_branch_count = 3
        self.artist_branch_count = 2

        self.frame_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, model_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.LayerNorm(model_dim),
        )

        frame_set_pool_dim = model_dim * (2 * frame_attention_heads + 2)
        self.frame_set_attentive_pool = MultiHeadAttentiveStatsPool(
            d_model=model_dim,
            num_heads=frame_attention_heads,
            hidden_dim=min(512, max(128, model_dim // 2)),
        )
        self.frame_set_projection = nn.Sequential(
            nn.LayerNorm(frame_set_pool_dim),
            nn.Linear(frame_set_pool_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim),
        )

        self.frame_cls_token = nn.Parameter(torch.empty(1, 1, model_dim))
        self.frame_pos_encoder = LearnedPositionalEncoding(
            d_model=model_dim,
            max_len=max_seq_len + 1,
            dropout=dropout,
        )
        frame_encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.frame_temporal_encoder = _make_transformer_encoder(
            encoder_layer=frame_encoder_layer,
            num_layers=frame_transformer_layers,
        )
        self.frame_temporal_attentive_pool = AttentiveStatsPool(
            d_model=model_dim,
            hidden_dim=min(256, max(64, model_dim // 2)),
        )
        self.frame_temporal_projection = nn.Sequential(
            nn.LayerNorm(model_dim * 5),
            nn.Linear(model_dim * 5, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim),
        )

        self.first_delta_attentive_pool = AttentiveStatsPool(
            d_model=model_dim,
            hidden_dim=min(256, max(64, model_dim // 2)),
        )
        self.second_delta_attentive_pool = AttentiveStatsPool(
            d_model=model_dim,
            hidden_dim=min(256, max(64, model_dim // 2)),
        )
        self.first_delta_projection = nn.Sequential(
            nn.LayerNorm(model_dim * 4),
            nn.Linear(model_dim * 4, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim),
        )
        self.second_delta_projection = nn.Sequential(
            nn.LayerNorm(model_dim * 4),
            nn.Linear(model_dim * 4, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim),
        )
        self.frame_delta_projection = nn.Sequential(
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim),
        )

        self.video_fusion_gate = nn.Sequential(
            nn.LayerNorm(model_dim * self.video_branch_count),
            nn.Linear(model_dim * self.video_branch_count, model_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(model_dim, model_dim * self.video_branch_count),
        )

        artist_set_pool_dim = model_dim * (2 * artist_attention_heads + 2)
        self.artist_set_attentive_pool = MaskedMultiHeadAttentiveStatsPool(
            d_model=model_dim,
            num_heads=artist_attention_heads,
            hidden_dim=min(512, max(128, model_dim // 2)),
        )
        self.artist_set_projection = nn.Sequential(
            nn.LayerNorm(artist_set_pool_dim),
            nn.Linear(artist_set_pool_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim),
        )

        artist_encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.artist_context_encoder = _make_transformer_encoder(
            encoder_layer=artist_encoder_layer,
            num_layers=artist_transformer_layers,
        )
        self.artist_context_attentive_pool = MaskedMultiHeadAttentiveStatsPool(
            d_model=model_dim,
            num_heads=artist_attention_heads,
            hidden_dim=min(512, max(128, model_dim // 2)),
        )
        self.artist_context_projection = nn.Sequential(
            nn.LayerNorm(artist_set_pool_dim),
            nn.Linear(artist_set_pool_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim),
        )

        self.artist_fusion_gate = nn.Sequential(
            nn.LayerNorm(model_dim * self.artist_branch_count),
            nn.Linear(model_dim * self.artist_branch_count, model_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(model_dim, model_dim * self.artist_branch_count),
        )

        self.projection_head = nn.Sequential(
            nn.LayerNorm(model_dim * (self.artist_branch_count + 1)),
            nn.Linear(model_dim * (self.artist_branch_count + 1), projection_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(projection_hidden_dim),
            nn.Linear(projection_hidden_dim, projection_mid_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.LayerNorm(projection_mid_dim),
            nn.Linear(projection_mid_dim, output_dim),
        )
        self.bnneck = SafeBatchNorm1d(output_dim)
        if self.bnneck.bias is not None:
            self.bnneck.bias.requires_grad_(False)

        self.raw_residual_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim),
            SafeBatchNorm1d(output_dim),
        )
        self.residual_gate = nn.Sequential(
            nn.LayerNorm(output_dim * 2),
            nn.Linear(output_dim * 2, output_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(output_dim, 1),
        )
        self.last_residual_alpha: Tensor | None = None
        self.last_video_mask: Tensor | None = None

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.frame_cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                _init_linear(module)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        final_gate = self.residual_gate[-1]
        if isinstance(final_gate, nn.Linear) and final_gate.bias is not None:
            nn.init.zeros_(final_gate.bias)

    @staticmethod
    def _pool_set_like(x: Tensor, attentive_pool: nn.Module) -> Tensor:
        attentive_stats = attentive_pool(x)
        mean_pool = x.mean(dim=1)
        max_pool = x.amax(dim=1)
        return torch.cat([attentive_stats, mean_pool, max_pool], dim=1)

    @staticmethod
    def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
        weights = mask.to(dtype=x.dtype).unsqueeze(-1)
        return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _masked_max(x: Tensor, mask: Tensor) -> Tensor:
        mask = mask.bool()
        if not mask.any(dim=1).all():
            mask = mask.clone()
            mask[~mask.any(dim=1), 0] = True
        masked = x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min)
        return masked.amax(dim=1)

    @staticmethod
    def _ensure_one_valid_video(mask: Tensor) -> Tensor:
        if mask.ndim != 2:
            raise ValueError(f"Expected video mask shape (batch, videos), got {tuple(mask.shape)}")
        mask = mask.bool()
        if mask.any(dim=1).all():
            return mask
        mask = mask.clone()
        mask[~mask.any(dim=1), 0] = True
        return mask

    def _prepare_artist_input(self, x: Tensor) -> tuple[Tensor, Tensor]:
        was_three_dim = x.ndim == 3
        if was_three_dim:
            x = x.unsqueeze(1)
        elif x.ndim != 4:
            raise ValueError(
                "Expected input shape (batch, frames, dim) or (batch, videos, frames, dim), "
                f"got {tuple(x.shape)}"
            )

        if x.size(1) <= 0 or x.size(2) <= 0:
            raise ValueError("Input must contain at least one video and one frame")
        if x.size(1) > self.max_videos:
            raise ValueError(f"Input video count {x.size(1)} exceeds max_videos {self.max_videos}")
        if x.size(2) > self.max_seq_len:
            raise ValueError(f"Input frame length {x.size(2)} exceeds max_seq_len {self.max_seq_len}")
        if x.size(3) != self.d_model:
            raise ValueError(f"Input feature dim {x.size(3)} does not match d_model {self.d_model}")

        x = x.float()
        if was_three_dim:
            video_mask = torch.ones(x.size(0), 1, dtype=torch.bool, device=x.device)
        else:
            video_mask = x.detach().abs().sum(dim=(2, 3)) > self.mask_eps
        return x, self._ensure_one_valid_video(video_mask)

    def _apply_video_dropout(self, video_mask: Tensor) -> Tensor:
        video_mask = self._ensure_one_valid_video(video_mask)
        if not self.training or self.video_dropout_p <= 0:
            return video_mask
        keep = torch.rand(video_mask.shape, device=video_mask.device) > self.video_dropout_p
        dropped_mask = video_mask & keep
        empty_rows = ~dropped_mask.any(dim=1)
        if empty_rows.any():
            first_valid = video_mask.float().argmax(dim=1)
            dropped_mask[empty_rows, first_valid[empty_rows]] = True
        return dropped_mask

    def _masked_frame_mean(self, x: Tensor, video_mask: Tensor) -> Tensor:
        weights = video_mask.to(dtype=x.dtype).view(x.size(0), x.size(1), 1, 1)
        summed = (x * weights).sum(dim=(1, 2))
        frame_count = int(x.size(2))
        denom = video_mask.to(dtype=x.dtype).sum(dim=1, keepdim=True).clamp_min(1.0) * frame_count
        return summed / denom

    def _raw_clip_residual(self, raw_x: Tensor, video_mask: Tensor) -> Tensor:
        raw_mean = self._masked_frame_mean(raw_x, video_mask)
        residual = self.raw_residual_projection(raw_mean)
        return F.normalize(residual, dim=1, eps=1e-8)

    def _encode_frame_set_branch(self, x: Tensor) -> Tensor:
        pooled = self._pool_set_like(x, self.frame_set_attentive_pool)
        return self.frame_set_projection(pooled)

    def _encode_frame_temporal_branch(self, x: Tensor) -> Tensor:
        cls = self.frame_cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, x], dim=1)
        tokens = self.frame_pos_encoder(tokens)
        tokens = self.frame_temporal_encoder(tokens)
        cls_out = tokens[:, 0]
        frame_tokens = tokens[:, 1:]
        attentive_stats = self.frame_temporal_attentive_pool(frame_tokens)
        mean_pool = frame_tokens.mean(dim=1)
        max_pool = frame_tokens.amax(dim=1)
        pooled = torch.cat([cls_out, attentive_stats, mean_pool, max_pool], dim=1)
        return self.frame_temporal_projection(pooled)

    def _encode_delta_order(self, deltas: Tensor, attentive_pool: AttentiveStatsPool, projection: nn.Module) -> Tensor:
        if deltas.size(1) <= 0:
            return deltas.new_zeros(deltas.size(0), self.model_dim)
        pooled = self._pool_set_like(deltas, attentive_pool)
        return projection(pooled)

    def _encode_frame_delta_branch(self, x: Tensor) -> Tensor:
        if x.size(1) < 2:
            first_delta_embedding = x.new_zeros(x.size(0), self.model_dim)
            second_delta_embedding = x.new_zeros(x.size(0), self.model_dim)
        else:
            first_deltas = x[:, 1:, :] - x[:, :-1, :]
            first_delta_embedding = self._encode_delta_order(
                first_deltas,
                self.first_delta_attentive_pool,
                self.first_delta_projection,
            )
            if first_deltas.size(1) < 2:
                second_delta_embedding = x.new_zeros(x.size(0), self.model_dim)
            else:
                second_deltas = first_deltas[:, 1:, :] - first_deltas[:, :-1, :]
                second_delta_embedding = self._encode_delta_order(
                    second_deltas,
                    self.second_delta_attentive_pool,
                    self.second_delta_projection,
                )
        return self.frame_delta_projection(torch.cat([first_delta_embedding, second_delta_embedding], dim=1))

    def _fuse_video_branches(self, branches: list[Tensor]) -> Tensor:
        branches = [F.normalize(branch, dim=1, eps=1e-8) for branch in branches]
        gate_input = torch.cat(branches, dim=1)
        branch_stack = torch.stack(branches, dim=1)
        gate_logits = self.video_fusion_gate(gate_input).view(
            gate_input.size(0), self.video_branch_count, self.model_dim
        )
        weights = torch.softmax(gate_logits, dim=1)
        return F.normalize(torch.sum(weights * branch_stack, dim=1), dim=1, eps=1e-8)

    def _encode_videos(self, frame_tokens: Tensor) -> Tensor:
        video_branches = [
            self._encode_frame_set_branch(frame_tokens),
            self._encode_frame_temporal_branch(frame_tokens),
            self._encode_frame_delta_branch(frame_tokens),
        ]
        return self._fuse_video_branches(video_branches)

    def _encode_artist_set_branch(self, video_tokens: Tensor, video_mask: Tensor) -> Tensor:
        attentive_stats = self.artist_set_attentive_pool(video_tokens, video_mask)
        mean_pool = self._masked_mean(video_tokens, video_mask)
        max_pool = self._masked_max(video_tokens, video_mask)
        return self.artist_set_projection(torch.cat([attentive_stats, mean_pool, max_pool], dim=1))

    def _encode_artist_context_branch(self, video_tokens: Tensor, video_mask: Tensor) -> Tensor:
        video_tokens = video_tokens * video_mask.to(dtype=video_tokens.dtype).unsqueeze(-1)
        contextualized = self.artist_context_encoder(video_tokens, src_key_padding_mask=~video_mask)
        attentive_stats = self.artist_context_attentive_pool(contextualized, video_mask)
        mean_pool = self._masked_mean(contextualized, video_mask)
        max_pool = self._masked_max(contextualized, video_mask)
        return self.artist_context_projection(torch.cat([attentive_stats, mean_pool, max_pool], dim=1))

    def _fuse_artist_branches(self, branches: list[Tensor]) -> tuple[Tensor, Tensor]:
        branches = [F.normalize(branch, dim=1, eps=1e-8) for branch in branches]
        gate_input = torch.cat(branches, dim=1)
        branch_stack = torch.stack(branches, dim=1)
        gate_logits = self.artist_fusion_gate(gate_input).view(
            gate_input.size(0), self.artist_branch_count, self.model_dim
        )
        weights = torch.softmax(gate_logits, dim=1)
        fused = F.normalize(torch.sum(weights * branch_stack, dim=1), dim=1, eps=1e-8)
        return fused, gate_input

    def encode(self, x: Tensor) -> Tensor:
        raw_x, video_mask = self._prepare_artist_input(x)
        video_mask = self._apply_video_dropout(video_mask)
        self.last_video_mask = video_mask.detach()

        batch_size, video_count, frame_count, _ = raw_x.shape
        raw_x = F.normalize(raw_x, dim=3, eps=1e-8)
        clip_residual = self._raw_clip_residual(raw_x, video_mask)

        frame_tokens = self.frame_projection(raw_x.reshape(batch_size * video_count, frame_count, self.d_model))
        video_tokens = self._encode_videos(frame_tokens).view(batch_size, video_count, self.model_dim)
        video_tokens = video_tokens * video_mask.to(dtype=video_tokens.dtype).unsqueeze(-1)

        artist_set_embedding = self._encode_artist_set_branch(video_tokens, video_mask)
        artist_context_embedding = self._encode_artist_context_branch(video_tokens, video_mask)
        artist_fused, artist_gate_input = self._fuse_artist_branches([artist_set_embedding, artist_context_embedding])

        fused = torch.cat([artist_fused, artist_gate_input], dim=1)
        learned_output = self.projection_head(fused)
        learned_output = self.bnneck(learned_output)
        learned_output = F.normalize(learned_output, dim=1, eps=1e-8)

        alpha = torch.sigmoid(self.residual_gate(torch.cat([clip_residual, learned_output], dim=1)))
        self.last_residual_alpha = alpha.detach()
        out = (1.0 - alpha) * clip_residual + alpha * learned_output
        return F.normalize(out, dim=1, eps=1e-8)


class ConvTripletNet(BaseTripletNet):
    def __init__(
        self,
        d_model: int = DEFAULT_EMBEDDING_DIM,
        seq_len: int = DEFAULT_MAX_SEQ_LEN,
        output_dim: int = DEFAULT_OUTPUT_DIM,
        conv_channels: list[int] | tuple[int, ...] = (256, 256, 128),
        kernel_sizes: list[int] | tuple[int, ...] = (3, 3, 3),
        strides: list[int] | tuple[int, ...] = (1, 1, 1),
        pool_after: set[int] | None = None,
        dropout: float = 0.3,
        hidden_dims: list[int] | tuple[int, ...] = (),
        global_pool: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0 or seq_len <= 0 or output_dim <= 0:
            raise ValueError("d_model, seq_len and output_dim must be positive")
        if not conv_channels:
            raise ValueError("conv_channels must not be empty")
        if not (len(conv_channels) == len(kernel_sizes) == len(strides)):
            raise ValueError("conv_channels, kernel_sizes, and strides must have the same length")

        pool_after = pool_after or set()
        layers: list[nn.Module] = []
        in_channels = d_model
        for idx, (out_channels, kernel_size, stride) in enumerate(zip(conv_channels, kernel_sizes, strides)):
            if out_channels <= 0 or kernel_size <= 0 or stride <= 0:
                raise ValueError("conv channels, kernel sizes and strides must be positive")
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
            nn.init.kaiming_normal_(conv.weight, nonlinearity="leaky_relu")
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
            layers.extend([
                conv,
                _make_group_norm(out_channels),
                nn.PReLU(num_parameters=out_channels),
                nn.Dropout(dropout),
            ])
            if idx in pool_after:
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True))
            in_channels = out_channels
        self.conv_stack = nn.Sequential(*layers)

        projection_layers: list[nn.Module] = []
        if global_pool:
            projection_layers.extend([nn.AdaptiveAvgPool1d(1), nn.Flatten()])
            prev_dim = in_channels
        else:
            fc_in_features = _infer_fc_in_features(self._forward_conv_only, seq_len=seq_len, d_model=d_model)
            projection_layers.append(nn.Flatten())
            prev_dim = fc_in_features

        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("hidden_dims must contain positive integers")
            linear = nn.Linear(prev_dim, hidden_dim)
            _init_linear(linear)
            projection_layers.extend([
                linear,
                nn.LayerNorm(hidden_dim),
                nn.PReLU(num_parameters=hidden_dim),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        final_linear = nn.Linear(prev_dim, output_dim)
        _init_linear(final_linear)
        projection_layers.append(final_linear)
        self.projection = nn.Sequential(*projection_layers)

    def _forward_conv_only(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, seq_len, dim), got {tuple(x.shape)}")
        return self.conv_stack(x.float().transpose(1, 2))

    def encode(self, x: Tensor) -> Tensor:
        x = self._forward_conv_only(x)
        x = self.projection(x)
        return F.normalize(x, dim=1, eps=1e-8)


class TripletNet1(BaseTripletNet):
    def __init__(
        self,
        d_model: int = DEFAULT_EMBEDDING_DIM,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        seq_len: int | None = None,
        output_dim: int = DEFAULT_OUTPUT_DIM,
        max_videos: int = DEFAULT_MAX_VIDEOS,
        video_dropout_p: float = 0.15,
        model_dim: int = 384,
        nhead: int | None = None,
        dim_feedforward: int | None = None,
        projection_hidden_dim: int | None = None,
        projection_mid_dim: int | None = None,
        frame_transformer_layers: int = 1,
        artist_transformer_layers: int = 1,
        dropout: float = 0.15,
        mask_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        max_seq_len = max_seq_len if seq_len is None else seq_len
        if d_model <= 0 or output_dim <= 0 or max_seq_len <= 0 or max_videos <= 0 or model_dim <= 0:
            raise ValueError("d_model, output_dim, max_seq_len, max_videos and model_dim must be positive")
        if frame_transformer_layers <= 0 or artist_transformer_layers <= 0:
            raise ValueError("transformer layer counts must be positive")
        if not 0.0 <= video_dropout_p < 1.0:
            raise ValueError("video_dropout_p must be in [0, 1)")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if mask_eps < 0:
            raise ValueError("mask_eps must be non-negative")

        nhead = _select_nhead(model_dim, preferred=8) if nhead is None else nhead
        if nhead <= 0 or model_dim % nhead != 0:
            raise ValueError(f"nhead={nhead} must be positive and divide model_dim={model_dim}")
        if dim_feedforward is None:
            dim_feedforward = 4 * model_dim
        if projection_hidden_dim is None:
            projection_hidden_dim = max(2 * model_dim, 2 * output_dim)
        if projection_mid_dim is None:
            projection_mid_dim = max(model_dim, output_dim)
        if projection_hidden_dim <= 0 or projection_mid_dim <= 0:
            raise ValueError("projection hidden dimensions must be positive")

        self.d_model = d_model
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len
        self.max_videos = max_videos
        self.video_dropout_p = float(video_dropout_p)
        self.mask_eps = float(mask_eps)

        self.frame_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, model_dim),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )
        self.frame_pos_encoder = LearnedPositionalEncoding(
            d_model=model_dim,
            max_len=max_seq_len,
            dropout=dropout,
        )
        frame_encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.frame_encoder = _make_transformer_encoder(
            encoder_layer=frame_encoder_layer,
            num_layers=frame_transformer_layers,
        )
        self.video_token_norm = nn.LayerNorm(model_dim)

        artist_encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.artist_encoder = _make_transformer_encoder(
            encoder_layer=artist_encoder_layer,
            num_layers=artist_transformer_layers,
        )
        self.artist_pool_norm = nn.LayerNorm(model_dim)

        self.projection_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, projection_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(projection_hidden_dim),
            nn.Linear(projection_hidden_dim, projection_mid_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.LayerNorm(projection_mid_dim),
            nn.Linear(projection_mid_dim, output_dim),
        )
        self.bnneck = SafeBatchNorm1d(output_dim)
        if self.bnneck.bias is not None:
            self.bnneck.bias.requires_grad_(False)

        self.last_video_mask: Tensor | None = None
        self.last_residual_alpha: Tensor | None = None
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                _init_linear(module)

    @staticmethod
    def _ensure_one_valid_video(mask: Tensor) -> Tensor:
        if mask.ndim != 2:
            raise ValueError(f"Expected video mask shape (batch, videos), got {tuple(mask.shape)}")
        mask = mask.bool()
        if mask.any(dim=1).all():
            return mask
        mask = mask.clone()
        mask[~mask.any(dim=1), 0] = True
        return mask

    @staticmethod
    def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
        weights = mask.to(dtype=x.dtype).unsqueeze(-1)
        return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _prepare_artist_input(self, x: Tensor) -> tuple[Tensor, Tensor]:
        was_three_dim = x.ndim == 3
        if was_three_dim:
            x = x.unsqueeze(1)
        elif x.ndim != 4:
            raise ValueError(
                "Expected input shape (batch, frames, dim) or (batch, videos, frames, dim), "
                f"got {tuple(x.shape)}"
            )

        if x.size(1) <= 0 or x.size(2) <= 0:
            raise ValueError("Input must contain at least one video and one frame")
        if x.size(1) > self.max_videos:
            raise ValueError(f"Input video count {x.size(1)} exceeds max_videos {self.max_videos}")
        if x.size(2) > self.max_seq_len:
            raise ValueError(f"Input frame length {x.size(2)} exceeds max_seq_len {self.max_seq_len}")
        if x.size(3) != self.d_model:
            raise ValueError(f"Input feature dim {x.size(3)} does not match d_model {self.d_model}")

        x = x.float()
        if was_three_dim:
            video_mask = torch.ones(x.size(0), 1, dtype=torch.bool, device=x.device)
        else:
            video_mask = x.detach().abs().sum(dim=(2, 3)) > self.mask_eps
        return x, self._ensure_one_valid_video(video_mask)

    def _apply_video_dropout(self, video_mask: Tensor) -> Tensor:
        video_mask = self._ensure_one_valid_video(video_mask)
        if not self.training or self.video_dropout_p <= 0:
            return video_mask
        keep = torch.rand(video_mask.shape, device=video_mask.device) > self.video_dropout_p
        dropped_mask = video_mask & keep
        empty_rows = ~dropped_mask.any(dim=1)
        if empty_rows.any():
            first_valid = video_mask.float().argmax(dim=1)
            dropped_mask[empty_rows, first_valid[empty_rows]] = True
        return dropped_mask

    def _encode_video_tokens(self, x: Tensor) -> Tensor:
        batch_size, video_count, frame_count, _ = x.shape
        frame_tokens = x.reshape(batch_size * video_count, frame_count, self.d_model)
        frame_tokens = self.frame_projection(frame_tokens)
        frame_tokens = self.frame_pos_encoder(frame_tokens)
        frame_tokens = self.frame_encoder(frame_tokens)
        video_tokens = frame_tokens.mean(dim=1)
        return self.video_token_norm(video_tokens).view(batch_size, video_count, self.model_dim)

    def encode(self, x: Tensor) -> Tensor:
        x, video_mask = self._prepare_artist_input(x)
        video_mask = self._apply_video_dropout(video_mask)
        self.last_video_mask = video_mask.detach()
        self.last_residual_alpha = None

        video_tokens = self._encode_video_tokens(x)
        video_tokens = video_tokens * video_mask.to(dtype=video_tokens.dtype).unsqueeze(-1)

        artist_tokens = self.artist_encoder(video_tokens, src_key_padding_mask=~video_mask)
        artist_tokens = artist_tokens * video_mask.to(dtype=artist_tokens.dtype).unsqueeze(-1)
        artist_embedding = self._masked_mean(artist_tokens, video_mask)
        artist_embedding = self.artist_pool_norm(artist_embedding)

        out = self.projection_head(artist_embedding)
        out = self.bnneck(out)
        return F.normalize(out, dim=1, eps=1e-8)


class TripletNet2(ConvTripletNet):
    def __init__(
        self,
        d_model: int = DEFAULT_EMBEDDING_DIM,
        seq_len: int = DEFAULT_MAX_SEQ_LEN,
        output_dim: int = DEFAULT_OUTPUT_DIM,
    ) -> None:
        super().__init__(
            d_model=d_model,
            seq_len=seq_len,
            output_dim=output_dim,
            conv_channels=(256, 256, 128),
            kernel_sizes=(3, 3, 3),
            strides=(1, 1, 1),
            pool_after={0, 1},
            hidden_dims=(256,),
        )


class TripletNet3(ConvTripletNet):
    def __init__(
        self,
        d_model: int = DEFAULT_EMBEDDING_DIM,
        seq_len: int = DEFAULT_MAX_SEQ_LEN,
        output_dim: int = DEFAULT_OUTPUT_DIM,
    ) -> None:
        super().__init__(
            d_model=d_model,
            seq_len=seq_len,
            output_dim=output_dim,
            conv_channels=(256, 256, 128, 128),
            kernel_sizes=(5, 5, 3, 3),
            strides=(1, 1, 1, 1),
            pool_after={0, 2},
            hidden_dims=(256,),
        )


class TripletNet4(ConvTripletNet):
    def __init__(
        self,
        d_model: int = DEFAULT_EMBEDDING_DIM,
        seq_len: int = DEFAULT_MAX_SEQ_LEN,
        output_dim: int = DEFAULT_OUTPUT_DIM,
    ) -> None:
        super().__init__(
            d_model=d_model,
            seq_len=seq_len,
            output_dim=output_dim,
            conv_channels=(256, 256, 128, 128),
            kernel_sizes=(3, 3, 3, 3),
            strides=(1, 2, 1, 2),
            pool_after=set(),
            hidden_dims=(256,),
        )


MODEL_REGISTRY: Dict[str, Type[BaseTripletNet]] = {
    "TripletNet1": TripletNet1,
    "TripletNet2": TripletNet2,
    "TripletNet3": TripletNet3,
    "TripletNet4": TripletNet4,
}


def build_model(name: str, **kwargs) -> BaseTripletNet:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model name: {name}. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
