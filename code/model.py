"""Model definitions for triplet learning on fixed-length CLIP frame embeddings.

The CLIP image encoder is assumed to be frozen before this file is used: the
training pipeline consumes precomputed per-frame CLIP embeddings, normally shaped
``(30, 768)`` for each video/artist representation.

TripletNet1 structure:
    input CLIP frame embeddings
    -> frame-level L2 normalisation
    -> shared 768 -> 512 projection adapter
    -> Set/Style attentive-statistics branch
    -> multi-scale temporal convolution branch
    -> 2-layer Temporal Transformer branch with positional encoding
    -> first/second-order temporal-delta branch
    -> CLIP mean residual anchor
    -> gated weighted-sum plus branch-concatenation fusion
    -> 1024 -> 512 -> 256 projection head
    -> final L2-normalised video/artist embedding

All models receive tensors shaped:
    (batch, seq_len, d_model)

All models return L2-normalised embeddings shaped:
    (batch, output_dim)
"""

from __future__ import annotations

from typing import Dict, Type

import torch
import torch.nn.functional as F
from torch import Tensor, nn


DEFAULT_EMBEDDING_DIM = 768
DEFAULT_OUTPUT_DIM = 256
DEFAULT_MAX_SEQ_LEN = 30


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


def _choose_transformer_dim(input_dim: int, preferred: int = 512) -> int:
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")

    if input_dim >= preferred:
        return preferred

    for divisor in (8, 4, 2, 1):
        candidate = input_dim - (input_dim % divisor)
        if candidate > 0:
            return candidate

    return input_dim


def _make_group_norm(num_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=1, num_channels=num_channels)


def _infer_fc_in_features(forward_fn, seq_len: int, d_model: int) -> int:
    if seq_len <= 0 or d_model <= 0:
        raise ValueError("seq_len and d_model must be positive")

    with torch.no_grad():
        dummy = torch.zeros(2, seq_len, d_model)
        out = forward_fn(dummy)
        return out.flatten(1).shape[1]


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
    """Attention-weighted mean + std pooling.

    This is stronger than plain mean pooling for retrieval because it keeps:
    1. which frames/posts are important;
    2. how dispersed the artist representation is across frames/posts.
    """

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


class BaseTripletNet(nn.Module):
    def encode(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def forward_once(self, x: Tensor) -> Tensor:
        return self.encode(x)

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.encode(anchor), self.encode(positive), self.encode(negative)


class HybridTransformerTripletNet(BaseTripletNet):
    def __init__(
        self,
        d_model: int = DEFAULT_EMBEDDING_DIM,
        output_dim: int = DEFAULT_OUTPUT_DIM,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        model_dim: int | None = None,
        nhead: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.15,
        dim_feedforward: int | None = None,
        projection_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()

        if d_model <= 0 or output_dim <= 0 or max_seq_len <= 0:
            raise ValueError("d_model, output_dim and max_seq_len must be positive")

        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        model_dim = _choose_transformer_dim(d_model, preferred=512) if model_dim is None else model_dim

        if model_dim <= 0:
            raise ValueError("model_dim must be positive")

        nhead = _select_nhead(model_dim, preferred=8) if nhead is None else nhead

        if nhead <= 0 or model_dim % nhead != 0:
            raise ValueError(f"nhead={nhead} must be positive and divide model_dim={model_dim}")

        self.d_model = d_model
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len

        self.input_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, model_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.LayerNorm(model_dim),
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, model_dim))

        self.pos_encoder = LearnedPositionalEncoding(
            d_model=model_dim,
            max_len=max_seq_len + 1,
            dropout=dropout,
        )

        if dim_feedforward is None:
            dim_feedforward = max(1024, 4 * model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.attentive_pool = AttentiveStatsPool(
            d_model=model_dim,
            hidden_dim=min(256, max(64, model_dim // 2)),
        )

        pooled_dim = model_dim * 5

        if projection_hidden_dim is None:
            projection_hidden_dim = max(512, output_dim * 2)

        mid_dim = max(output_dim, projection_hidden_dim // 2)

        self.projection = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, projection_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_hidden_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(mid_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                _init_linear(module)

    def encode(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, seq_len, dim), got {tuple(x.shape)}")

        if x.size(1) > self.max_seq_len:
            raise ValueError(f"Input sequence length {x.size(1)} exceeds max_seq_len {self.max_seq_len}")

        if x.size(2) != self.d_model:
            raise ValueError(f"Input feature dim {x.size(2)} does not match d_model {self.d_model}")

        x = self.input_projection(x.float())

        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        cls_out = x[:, 0]
        frame_tokens = x[:, 1:]

        attentive_stats = self.attentive_pool(frame_tokens)
        mean_pool = frame_tokens.mean(dim=1)
        max_pool = frame_tokens.amax(dim=1)

        pooled = torch.cat(
            [
                cls_out,
                attentive_stats,
                mean_pool,
                max_pool,
            ],
            dim=1,
        )

        x = self.projection(pooled)

        return F.normalize(x, dim=1, eps=1e-8)



class DualBranchCLIPTripletNet(BaseTripletNet):
    """Dual-branch aggregator for precomputed CLIP frame embeddings.

    The CLIP encoder is not trained here. It is expected to have been used offline
    to produce a sequence of frame embeddings. This network learns only how to
    aggregate those frozen frame embeddings into a retrieval-ready representation.
    """

    def __init__(
        self,
        d_model: int = DEFAULT_EMBEDDING_DIM,
        output_dim: int = DEFAULT_OUTPUT_DIM,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        model_dim: int = 384,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.15,
        dim_feedforward: int | None = None,
        projection_hidden_dim: int = 512,
        use_delta_branch: bool = True,
    ) -> None:
        super().__init__()

        if d_model <= 0 or output_dim <= 0 or max_seq_len <= 0 or model_dim <= 0:
            raise ValueError("d_model, output_dim, max_seq_len and model_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if nhead <= 0 or model_dim % nhead != 0:
            raise ValueError(f"nhead={nhead} must be positive and divide model_dim={model_dim}")
        if projection_hidden_dim <= 0:
            raise ValueError("projection_hidden_dim must be positive")

        self.d_model = d_model
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len
        self.use_delta_branch = use_delta_branch

        # Shared CLIP-frame adapter: 768 -> 384 by default.
        # The input is first L2-normalised per frame because CLIP embeddings are
        # usually compared by cosine similarity.
        self.frame_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, model_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.LayerNorm(model_dim),
        )

        # Branch A: set/style branch. No positional encoding is used here so the
        # model can learn the order-invariant visual/content style distribution of
        # the sampled frames.
        self.set_attentive_pool = AttentiveStatsPool(
            d_model=model_dim,
            hidden_dim=min(256, max(64, model_dim // 2)),
        )
        self.set_projection = nn.Sequential(
            nn.LayerNorm(model_dim * 4),
            nn.Linear(model_dim * 4, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim),
        )

        # Branch B: temporal branch. This keeps the equal-interval frame order.
        self.cls_token = nn.Parameter(torch.empty(1, 1, model_dim))
        self.pos_encoder = LearnedPositionalEncoding(
            d_model=model_dim,
            max_len=max_seq_len + 1,
            dropout=dropout,
        )

        if dim_feedforward is None:
            dim_feedforward = 4 * model_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        self.temporal_attentive_pool = AttentiveStatsPool(
            d_model=model_dim,
            hidden_dim=min(256, max(64, model_dim // 2)),
        )
        self.temporal_projection = nn.Sequential(
            nn.LayerNorm(model_dim * 5),
            nn.Linear(model_dim * 5, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim),
        )

        # Optional branch C: temporal delta. It encodes semantic changes between
        # adjacent sampled frames, which is useful when the video contains scene
        # changes, motion cues, editing rhythm, or narrative progression.
        if use_delta_branch:
            self.delta_attentive_pool = AttentiveStatsPool(
                d_model=model_dim,
                hidden_dim=min(256, max(64, model_dim // 2)),
            )
            self.delta_projection = nn.Sequential(
                nn.LayerNorm(model_dim * 4),
                nn.Linear(model_dim * 4, model_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(model_dim),
            )
            gate_input_dim = model_dim * 3
            gate_outputs = 3
        else:
            self.delta_attentive_pool = None
            self.delta_projection = None
            gate_input_dim = model_dim * 2
            gate_outputs = 2

        # Gated fusion learns whether each sample should rely more on global style,
        # temporal ordering, or frame-to-frame semantic change.
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(model_dim, gate_outputs),
        )

        # Projection head: 384 -> 512 -> 256 by default.
        self.projection_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, projection_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                _init_linear(module)

    @staticmethod
    def _pool_set_like(x: Tensor, attentive_pool: AttentiveStatsPool) -> Tensor:
        attentive_stats = attentive_pool(x)
        mean_pool = x.mean(dim=1)
        max_pool = x.amax(dim=1)
        return torch.cat([attentive_stats, mean_pool, max_pool], dim=1)

    def _encode_set_branch(self, x: Tensor) -> Tensor:
        pooled = self._pool_set_like(x, self.set_attentive_pool)
        return self.set_projection(pooled)

    def _encode_temporal_branch(self, x: Tensor) -> Tensor:
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, x], dim=1)
        tokens = self.pos_encoder(tokens)
        tokens = self.temporal_encoder(tokens)

        cls_out = tokens[:, 0]
        frame_tokens = tokens[:, 1:]
        attentive_stats = self.temporal_attentive_pool(frame_tokens)
        mean_pool = frame_tokens.mean(dim=1)
        max_pool = frame_tokens.amax(dim=1)
        pooled = torch.cat([cls_out, attentive_stats, mean_pool, max_pool], dim=1)
        return self.temporal_projection(pooled)

    def _encode_delta_branch(self, x: Tensor) -> Tensor:
        if not self.use_delta_branch or self.delta_attentive_pool is None or self.delta_projection is None:
            raise RuntimeError("delta branch is disabled")

        if x.size(1) < 2:
            return x.new_zeros(x.size(0), self.model_dim)

        deltas = x[:, 1:, :] - x[:, :-1, :]
        pooled = self._pool_set_like(deltas, self.delta_attentive_pool)
        return self.delta_projection(pooled)

    def encode(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, seq_len, dim), got {tuple(x.shape)}")
        if x.size(1) > self.max_seq_len:
            raise ValueError(f"Input sequence length {x.size(1)} exceeds max_seq_len {self.max_seq_len}")
        if x.size(2) != self.d_model:
            raise ValueError(f"Input feature dim {x.size(2)} does not match d_model {self.d_model}")

        # Frozen CLIP frame embeddings are already precomputed. This layer performs
        # frame-level cosine normalisation before the learnable aggregation network.
        x = F.normalize(x.float(), dim=2, eps=1e-8)
        x = self.frame_projection(x)

        set_embedding = self._encode_set_branch(x)
        temporal_embedding = self._encode_temporal_branch(x)

        if self.use_delta_branch:
            delta_embedding = self._encode_delta_branch(x)
            gate_input = torch.cat([set_embedding, temporal_embedding, delta_embedding], dim=1)
            weights = torch.softmax(self.fusion_gate(gate_input), dim=1)
            fused = (
                weights[:, 0:1] * set_embedding
                + weights[:, 1:2] * temporal_embedding
                + weights[:, 2:3] * delta_embedding
            )
        else:
            gate_input = torch.cat([set_embedding, temporal_embedding], dim=1)
            weights = torch.softmax(self.fusion_gate(gate_input), dim=1)
            fused = weights[:, 0:1] * set_embedding + weights[:, 1:2] * temporal_embedding

        x = self.projection_head(fused)
        return F.normalize(x, dim=1, eps=1e-8)


class EnhancedCLIPRetrievalTripletNet(BaseTripletNet):
    """Four-branch CLIP-frame aggregator tuned for artist retrieval.

    A deterministic CLIP-mean residual is blended into the learned representation
    so early epochs keep a stable retrieval prior instead of relying entirely on
    a randomly initialised deep projection head.
    """

    def __init__(
        self,
        d_model: int = DEFAULT_EMBEDDING_DIM,
        output_dim: int = DEFAULT_OUTPUT_DIM,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        model_dim: int = 512,
        nhead: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.15,
        dim_feedforward: int | None = None,
        projection_hidden_dim: int = 1024,
        projection_mid_dim: int = 512,
        conv_kernel_sizes: tuple[int, ...] = (3, 5, 7),
        conv_channels: int | None = None,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()

        if d_model <= 0 or output_dim <= 0 or max_seq_len <= 0 or model_dim <= 0:
            raise ValueError("d_model, output_dim, max_seq_len and model_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if projection_hidden_dim <= 0 or projection_mid_dim <= 0:
            raise ValueError("projection hidden dimensions must be positive")
        if not conv_kernel_sizes:
            raise ValueError("conv_kernel_sizes must not be empty")
        if any(kernel_size <= 0 for kernel_size in conv_kernel_sizes):
            raise ValueError("conv_kernel_sizes must contain positive integers")
        if attention_heads <= 0:
            raise ValueError("attention_heads must be positive")

        nhead = _select_nhead(model_dim, preferred=8) if nhead is None else nhead
        if nhead <= 0 or model_dim % nhead != 0:
            raise ValueError(f"nhead={nhead} must be positive and divide model_dim={model_dim}")

        conv_channels = conv_channels or max(128, model_dim // 2)
        if conv_channels <= 0:
            raise ValueError("conv_channels must be positive")

        self.d_model = d_model
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len
        self.learned_output_logit = nn.Parameter(torch.tensor(-1.0))

        self.frame_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, model_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.LayerNorm(model_dim),
        )

        set_pool_dim = model_dim * (2 * attention_heads + 2)
        self.set_attentive_pool = MultiHeadAttentiveStatsPool(
            d_model=model_dim,
            num_heads=attention_heads,
            hidden_dim=min(512, max(128, model_dim // 2)),
        )
        self.set_projection = nn.Sequential(
            nn.LayerNorm(set_pool_dim),
            nn.Linear(set_pool_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim),
        )

        self.conv_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(model_dim, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                    _make_group_norm(conv_channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                    _make_group_norm(conv_channels),
                    nn.GELU(),
                )
                for kernel_size in conv_kernel_sizes
            ]
        )
        conv_pool_dim = len(conv_kernel_sizes) * conv_channels * 2
        self.conv_projection = nn.Sequential(
            nn.LayerNorm(conv_pool_dim),
            nn.Linear(conv_pool_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim),
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, model_dim))
        self.pos_encoder = LearnedPositionalEncoding(
            d_model=model_dim,
            max_len=max_seq_len + 1,
            dropout=dropout,
        )
        if dim_feedforward is None:
            dim_feedforward = 4 * model_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        self.temporal_attentive_pool = AttentiveStatsPool(
            d_model=model_dim,
            hidden_dim=min(256, max(64, model_dim // 2)),
        )
        self.temporal_projection = nn.Sequential(
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
        self.delta_projection = nn.Sequential(
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim),
        )

        self.branch_count = 4
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(model_dim * self.branch_count),
            nn.Linear(model_dim * self.branch_count, model_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(model_dim, self.branch_count),
        )
        self.projection_head = nn.Sequential(
            nn.LayerNorm(model_dim * (self.branch_count + 1)),
            nn.Linear(model_dim * (self.branch_count + 1), projection_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(projection_hidden_dim),
            nn.Linear(projection_hidden_dim, projection_mid_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.LayerNorm(projection_mid_dim),
            nn.Linear(projection_mid_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                _init_linear(module)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _pool_set_like(x: Tensor, attentive_pool: nn.Module) -> Tensor:
        attentive_stats = attentive_pool(x)
        mean_pool = x.mean(dim=1)
        max_pool = x.amax(dim=1)
        return torch.cat([attentive_stats, mean_pool, max_pool], dim=1)

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, seq_len, dim), got {tuple(x.shape)}")
        if x.size(1) <= 0:
            raise ValueError("Input sequence length must be positive")
        if x.size(1) > self.max_seq_len:
            raise ValueError(f"Input sequence length {x.size(1)} exceeds max_seq_len {self.max_seq_len}")
        if x.size(2) != self.d_model:
            raise ValueError(f"Input feature dim {x.size(2)} does not match d_model {self.d_model}")

    def _raw_clip_residual(self, x: Tensor) -> Tensor:
        raw_mean = x.mean(dim=1)
        if self.output_dim == self.d_model:
            residual = raw_mean
        elif self.output_dim < self.d_model:
            residual = F.adaptive_avg_pool1d(raw_mean.unsqueeze(1), self.output_dim).squeeze(1)
        else:
            residual = F.pad(raw_mean, (0, self.output_dim - self.d_model))
        return F.normalize(residual, dim=1, eps=1e-8)

    def _encode_set_branch(self, x: Tensor) -> Tensor:
        pooled = self._pool_set_like(x, self.set_attentive_pool)
        return self.set_projection(pooled)

    def _encode_conv_branch(self, x: Tensor) -> Tensor:
        tokens = x.transpose(1, 2)
        pooled_features: list[Tensor] = []
        for block in self.conv_blocks:
            features = block(tokens)
            pooled_features.append(F.adaptive_avg_pool1d(features, output_size=1).squeeze(-1))
            pooled_features.append(F.adaptive_max_pool1d(features, output_size=1).squeeze(-1))
        return self.conv_projection(torch.cat(pooled_features, dim=1))

    def _encode_temporal_branch(self, x: Tensor) -> Tensor:
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, x], dim=1)
        tokens = self.pos_encoder(tokens)
        tokens = self.temporal_encoder(tokens)

        cls_out = tokens[:, 0]
        frame_tokens = tokens[:, 1:]
        attentive_stats = self.temporal_attentive_pool(frame_tokens)
        mean_pool = frame_tokens.mean(dim=1)
        max_pool = frame_tokens.amax(dim=1)
        pooled = torch.cat([cls_out, attentive_stats, mean_pool, max_pool], dim=1)
        return self.temporal_projection(pooled)

    def _encode_delta_order(
        self,
        deltas: Tensor,
        attentive_pool: AttentiveStatsPool,
        projection: nn.Module,
    ) -> Tensor:
        if deltas.size(1) <= 0:
            return deltas.new_zeros(deltas.size(0), self.model_dim)
        pooled = self._pool_set_like(deltas, attentive_pool)
        return projection(pooled)

    def _encode_delta_branch(self, x: Tensor) -> Tensor:
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

        return self.delta_projection(torch.cat([first_delta_embedding, second_delta_embedding], dim=1))

    def encode(self, x: Tensor) -> Tensor:
        self._validate_input(x)

        x = F.normalize(x.float(), dim=2, eps=1e-8)
        clip_residual = self._raw_clip_residual(x)
        x = self.frame_projection(x)

        branch_embeddings = [
            self._encode_set_branch(x),
            self._encode_conv_branch(x),
            self._encode_temporal_branch(x),
            self._encode_delta_branch(x),
        ]
        branch_embeddings = [F.normalize(branch, dim=1, eps=1e-8) for branch in branch_embeddings]
        gate_input = torch.cat(branch_embeddings, dim=1)
        branch_stack = torch.stack(branch_embeddings, dim=1)
        weights = torch.softmax(self.fusion_gate(gate_input), dim=1).unsqueeze(-1)
        weighted_sum = torch.sum(weights * branch_stack, dim=1)

        fused = torch.cat([weighted_sum, gate_input], dim=1)
        learned_output = F.normalize(self.projection_head(fused), dim=1, eps=1e-8)
        learned_weight = torch.sigmoid(self.learned_output_logit)
        x = (1.0 - learned_weight) * clip_residual + learned_weight * learned_output
        return F.normalize(x, dim=1, eps=1e-8)


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

        for idx, (out_channels, kernel_size, stride) in enumerate(
            zip(conv_channels, kernel_sizes, strides)
        ):
            if out_channels <= 0 or kernel_size <= 0 or stride <= 0:
                raise ValueError("conv channels, kernel sizes and strides must be positive")

            padding = kernel_size // 2

            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )

            nn.init.kaiming_normal_(conv.weight, nonlinearity="leaky_relu")

            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

            layers.extend(
                [
                    conv,
                    _make_group_norm(out_channels),
                    nn.PReLU(num_parameters=out_channels),
                    nn.Dropout(dropout),
                ]
            )

            if idx in pool_after:
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True))

            in_channels = out_channels

        self.conv_stack = nn.Sequential(*layers)

        projection_layers: list[nn.Module] = []

        if global_pool:
            projection_layers.extend(
                [
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                ]
            )
            prev_dim = in_channels
        else:
            fc_in_features = _infer_fc_in_features(
                self._forward_conv_only,
                seq_len=seq_len,
                d_model=d_model,
            )
            projection_layers.append(nn.Flatten())
            prev_dim = fc_in_features

        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("hidden_dims must contain positive integers")

            linear = nn.Linear(prev_dim, hidden_dim)
            _init_linear(linear)

            projection_layers.extend(
                [
                    linear,
                    nn.LayerNorm(hidden_dim),
                    nn.PReLU(num_parameters=hidden_dim),
                    nn.Dropout(dropout),
                ]
            )

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


class TripletNet1(EnhancedCLIPRetrievalTripletNet):
    def __init__(
        self,
        d_model: int = DEFAULT_EMBEDDING_DIM,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        seq_len: int | None = None,
        output_dim: int = DEFAULT_OUTPUT_DIM,
    ) -> None:
        super().__init__(
            d_model=d_model,
            max_seq_len=max_seq_len if seq_len is None else seq_len,
            output_dim=output_dim,
            model_dim=512,
            nhead=8,
            num_layers=2,
            dropout=0.15,
            dim_feedforward=2048,
            projection_hidden_dim=1024,
            projection_mid_dim=512,
            conv_kernel_sizes=(3, 5, 7),
            attention_heads=4,
        )


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


class TripletNet5(ConvTripletNet):
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
            conv_channels=(256, 256, 256, 128, 128),
            kernel_sizes=(3, 3, 3, 3, 3),
            strides=(1, 1, 2, 1, 2),
            pool_after={0},
            hidden_dims=(512, 256),
        )


MODEL_REGISTRY: Dict[str, Type[BaseTripletNet]] = {
    "TripletNet1": TripletNet1,
    "TripletNet2": TripletNet2,
    "TripletNet3": TripletNet3,
    "TripletNet4": TripletNet4,
    "TripletNet5": TripletNet5,
}


def build_model(name: str, **kwargs) -> BaseTripletNet:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model name: {name}. Available: {sorted(MODEL_REGISTRY)}")

    return MODEL_REGISTRY[name](**kwargs)
