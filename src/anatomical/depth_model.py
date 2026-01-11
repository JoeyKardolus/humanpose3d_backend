#!/usr/bin/env python3
"""
PoseFormer Transformer for depth correction.

GPU-optimized architecture for RTX 5080:
- Temporal-spatial attention
- FP16 mixed precision support
- Batch size 256
- Predicts depth corrections (Δz) per marker
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1)]


class TemporalAttention(nn.Module):
    """Multi-head temporal attention across frames."""

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply multi-head attention.

        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Attention output (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, v)

        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(context)

        return output


class SpatialAttention(nn.Module):
    """Multi-head spatial attention across markers."""

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.attention = TemporalAttention(d_model, num_heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention across markers.

        Args:
            x: (batch, frames, markers, d_model)

        Returns:
            Attention output (batch, frames, markers, d_model)
        """
        batch_size, frames, markers, d_model = x.shape

        # Process each frame independently
        outputs = []
        for t in range(frames):
            frame = x[:, t, :, :]  # (batch, markers, d_model)
            attn_out = self.attention(frame)
            outputs.append(attn_out)

        return torch.stack(outputs, dim=1)  # (batch, frames, markers, d_model)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Transformer block with temporal and spatial attention."""

    def __init__(self, d_model: int, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()

        # Temporal attention
        self.temporal_attn = TemporalAttention(d_model, num_heads, dropout)
        self.temporal_norm = nn.LayerNorm(d_model)

        # Spatial attention
        self.spatial_attn = SpatialAttention(d_model, num_heads, dropout)
        self.spatial_norm = nn.LayerNorm(d_model)

        # Feed-forward
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ff_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer block.

        Args:
            x: (batch, frames, markers, d_model)

        Returns:
            Transformed features (batch, frames, markers, d_model)
        """
        batch_size, frames, markers, d_model = x.shape

        # Temporal attention (across frames for each marker)
        x_reshaped = x.permute(0, 2, 1, 3).contiguous()  # (batch, markers, frames, d_model)
        x_reshaped = x_reshaped.view(batch_size * markers, frames, d_model)

        temporal_out = self.temporal_attn(x_reshaped)
        temporal_out = x_reshaped + self.dropout(temporal_out)
        temporal_out = self.temporal_norm(temporal_out)

        temporal_out = temporal_out.view(batch_size, markers, frames, d_model)
        temporal_out = temporal_out.permute(0, 2, 1, 3).contiguous()  # (batch, frames, markers, d_model)

        # Spatial attention (across markers for each frame)
        spatial_out = self.spatial_attn(temporal_out)
        spatial_out = temporal_out + self.dropout(spatial_out)
        spatial_out = self.spatial_norm(spatial_out)

        # Feed-forward
        ff_out = self.ff(spatial_out)
        output = spatial_out + self.dropout(ff_out)
        output = self.ff_norm(output)

        return output


class PoseFormerDepthRefiner(nn.Module):
    """PoseFormer Transformer for depth correction.

    Architecture:
    - Input: (batch, frames, markers, features)
    - Embedding layer
    - N Transformer blocks (temporal + spatial attention)
    - Output heads: depth correction (Δz) + confidence

    Optimized for RTX 5080:
    - FP16 mixed precision support
    - Efficient attention implementation
    - Batch size 256
    """

    def __init__(
        self,
        num_markers: int = 59,
        num_frames: int = 11,
        feature_dim: int = 7,  # x, y, z, visibility, variance, is_augmented, marker_type
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_markers = num_markers
        self.num_frames = num_frames
        self.d_model = d_model

        # Input embedding
        self.input_projection = nn.Linear(feature_dim, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=num_frames)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output heads
        self.depth_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input features (batch, frames, markers, feature_dim)

        Returns:
            delta_z: Depth corrections (batch, frames, markers)
            confidence: Confidence scores (batch, frames, markers)
        """
        batch_size, frames, markers, _ = x.shape

        # Project to d_model
        x = self.input_projection(x)  # (batch, frames, markers, d_model)

        # Add positional encoding (frame-wise)
        for m in range(markers):
            x[:, :, m, :] = self.pos_encoding(x[:, :, m, :])

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Extract center frame features (or all frames for sequence output)
        # For now, predict for all frames

        # Output predictions
        delta_z = self.depth_head(x).squeeze(-1)  # (batch, frames, markers)
        confidence = self.confidence_head(x).squeeze(-1)  # (batch, frames, markers)

        return delta_z, confidence

    def predict_single_frame(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict depth correction for center frame only.

        Args:
            x: Input features (batch, frames, markers, feature_dim)

        Returns:
            delta_z: Depth corrections for center frame (batch, markers)
            confidence: Confidence scores for center frame (batch, markers)
        """
        delta_z_all, confidence_all = self.forward(x)

        center_idx = x.size(1) // 2
        return delta_z_all[:, center_idx, :], confidence_all[:, center_idx, :]


def create_model(device: str = "cuda") -> PoseFormerDepthRefiner:
    """Create PoseFormer model optimized for RTX 5080.

    Args:
        device: Device to place model on ("cuda" or "cpu")

    Returns:
        Initialized model
    """
    model = PoseFormerDepthRefiner(
        num_markers=59,
        num_frames=11,
        feature_dim=7,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        dropout=0.1,
    )

    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"PoseFormer Model:")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {num_trainable:,}")
    print(f"  Device: {device}")

    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing PoseFormer architecture...")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(device)

    # Test forward pass
    batch_size = 4
    frames = 11
    markers = 59
    features = 7

    x = torch.randn(batch_size, frames, markers, features).to(device)

    print()
    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        delta_z, confidence = model(x)

    print(f"Output delta_z shape: {delta_z.shape}")
    print(f"Output confidence shape: {confidence.shape}")
    print()
    print(f"Sample delta_z: {delta_z[0, 5, :3]}")
    print(f"Sample confidence: {confidence[0, 5, :3]}")
    print()
    print("✓ Model working correctly!")
