"""Camera-space POF prediction neural network.

Predicts Part Orientation Fields (POF) unit vectors from normalized 2D keypoints
and 2D foreshortening features. Operates entirely in camera space.

Key insight from MTC: 2D foreshortening (how short a limb appears in 2D)
directly encodes 3D limb orientation. A limb pointing at the camera appears
short in 2D; one parallel to the image plane appears at full length.

Architecture:
1. LimbEncoder: Per-limb features from 2D foreshortening + joint positions
2. LimbTransformer: Inter-limb reasoning via multi-head attention
3. POFHead: Per-limb MLP producing unit vectors
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .constants import NUM_JOINTS, NUM_LIMBS, LIMB_DEFINITIONS


class LimbEncoder(nn.Module):
    """Encode per-limb features from 2D foreshortening and joint positions.

    Input features per limb:
    - limb_delta_2d: (14, 2) normalized 2D displacement direction
    - limb_length_2d: (14,) 2D length (foreshortening magnitude)
    - parent_pos: (14, 2) parent joint 2D position
    - child_pos: (14, 2) child joint 2D position
    - parent_vis: (14,) parent joint visibility
    - child_vis: (14,) child joint visibility

    Total per-limb: 2 + 1 + 2 + 2 + 1 + 1 = 9 features
    """

    def __init__(
        self,
        d_model: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Per-limb input: 9 features
        self.limb_input_dim = 9

        # Encode per-limb features
        self.limb_encoder = nn.Sequential(
            nn.Linear(self.limb_input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Global context from full pose (17 joints x 2 coords + 17 vis = 51)
        self.global_input_dim = NUM_JOINTS * 2 + NUM_JOINTS
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Combine limb features with global context
        self.combine = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(
        self,
        pose_2d: torch.Tensor,       # (batch, 17, 2)
        visibility: torch.Tensor,     # (batch, 17)
        limb_delta_2d: torch.Tensor,  # (batch, 14, 2)
        limb_length_2d: torch.Tensor, # (batch, 14)
    ) -> torch.Tensor:
        """Encode per-limb features.

        Returns:
            (batch, 14, d_model) per-limb encoded features
        """
        batch_size = pose_2d.size(0)
        device = pose_2d.device

        # Extract per-limb joint features
        parent_pos = torch.zeros(batch_size, NUM_LIMBS, 2, device=device)
        child_pos = torch.zeros(batch_size, NUM_LIMBS, 2, device=device)
        parent_vis = torch.zeros(batch_size, NUM_LIMBS, device=device)
        child_vis = torch.zeros(batch_size, NUM_LIMBS, device=device)

        for limb_idx, (parent, child) in enumerate(LIMB_DEFINITIONS):
            parent_pos[:, limb_idx] = pose_2d[:, parent]
            child_pos[:, limb_idx] = pose_2d[:, child]
            parent_vis[:, limb_idx] = visibility[:, parent]
            child_vis[:, limb_idx] = visibility[:, child]

        # Concatenate per-limb features: (batch, 14, 9)
        limb_features = torch.cat([
            limb_delta_2d,                      # (batch, 14, 2)
            limb_length_2d.unsqueeze(-1),       # (batch, 14, 1)
            parent_pos,                         # (batch, 14, 2)
            child_pos,                          # (batch, 14, 2)
            parent_vis.unsqueeze(-1),           # (batch, 14, 1)
            child_vis.unsqueeze(-1),            # (batch, 14, 1)
        ], dim=-1)

        # Encode per-limb features
        limb_encoded = self.limb_encoder(limb_features)  # (batch, 14, d_model)

        # Global context from full pose
        pose_flat = pose_2d.reshape(batch_size, -1)  # (batch, 34)
        global_features = torch.cat([pose_flat, visibility], dim=-1)  # (batch, 51)
        global_encoded = self.global_encoder(global_features)  # (batch, d_model)

        # Broadcast global context to each limb and combine
        global_broadcast = global_encoded.unsqueeze(1).expand(-1, NUM_LIMBS, -1)
        combined = torch.cat([limb_encoded, global_broadcast], dim=-1)
        output = self.combine(combined)  # (batch, 14, d_model)

        return output


class LimbTransformer(nn.Module):
    """Transformer for inter-limb reasoning.

    Uses multi-head self-attention for limb-to-limb communication.
    This allows the model to enforce anatomical constraints (e.g.,
    symmetric arms, consistent torso orientation).
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_limbs: int = NUM_LIMBS,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_limbs = num_limbs

        # Learnable positional encoding for limb identity
        self.pos_encoding = nn.Parameter(
            torch.randn(num_limbs, d_model) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, limb_features: torch.Tensor) -> torch.Tensor:
        """Process limb features through transformer.

        Args:
            limb_features: (batch, 14, d_model) from LimbEncoder

        Returns:
            (batch, 14, d_model) refined per-limb features
        """
        # Add positional encoding for limb identity
        x = limb_features + self.pos_encoding.unsqueeze(0)

        # Apply transformer
        return self.transformer(x)  # (batch, 14, d_model)


class POFHead(nn.Module):
    """Predict POF unit vectors for each limb.

    Uses shared initial layer + per-limb specialized heads.
    Output is normalized to unit vectors.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_limbs: int = NUM_LIMBS,
    ):
        super().__init__()
        self.num_limbs = num_limbs

        # Shared initial processing
        self.shared = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
        )

        # Per-limb final projection to 3D vector
        self.limb_heads = nn.ModuleList([
            nn.Linear(d_model // 2, 3)
            for _ in range(num_limbs)
        ])

        # Initialize to small weights for stable training
        for head in self.limb_heads:
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)

    def forward(self, limb_features: torch.Tensor) -> torch.Tensor:
        """Predict POF unit vectors.

        Args:
            limb_features: (batch, 14, d_model)

        Returns:
            (batch, 14, 3) unit vectors (normalized)
        """
        batch_size = limb_features.size(0)

        # Shared processing
        shared_out = self.shared(limb_features)  # (batch, 14, d_model//2)

        # Per-limb predictions
        predictions = []
        for i, head in enumerate(self.limb_heads):
            vec = head(shared_out[:, i, :])  # (batch, 3)
            predictions.append(vec)

        pof = torch.stack(predictions, dim=1)  # (batch, 14, 3)

        # Normalize to unit vectors
        pof = F.normalize(pof, dim=-1, eps=1e-6)

        return pof


class CameraPOFModel(nn.Module):
    """Complete camera-space POF prediction model.

    Predicts 14 POF unit vectors from normalized 2D pose and foreshortening features.
    Operates entirely in camera space without world transformations.

    Architecture:
    1. LimbEncoder: Per-limb features from 2D foreshortening + positions
    2. LimbTransformer: Inter-limb attention for anatomical consistency
    3. POFHead: Per-limb unit vector prediction

    Input:
    - pose_2d: (batch, 17, 2) normalized 2D coordinates (pelvis-centered, unit-torso)
    - visibility: (batch, 17) per-joint visibility scores
    - limb_delta_2d: (batch, 14, 2) normalized 2D displacement directions
    - limb_length_2d: (batch, 14) 2D lengths (foreshortening magnitude)

    Output:
    - pof: (batch, 14, 3) unit vectors for each limb
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.encoder = LimbEncoder(d_model, dropout=dropout)
        self.transformer = LimbTransformer(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.pof_head = POFHead(d_model)

    def forward(
        self,
        pose_2d: torch.Tensor,        # (batch, 17, 2)
        visibility: torch.Tensor,      # (batch, 17)
        limb_delta_2d: torch.Tensor,   # (batch, 14, 2)
        limb_length_2d: torch.Tensor,  # (batch, 14)
    ) -> torch.Tensor:
        """Predict POF unit vectors.

        Args:
            pose_2d: (batch, 17, 2) normalized 2D coordinates
            visibility: (batch, 17) per-joint visibility scores
            limb_delta_2d: (batch, 14, 2) normalized 2D displacement directions
            limb_length_2d: (batch, 14) 2D lengths

        Returns:
            (batch, 14, 3) unit vectors for each limb
        """
        # Encode per-limb features
        limb_features = self.encoder(pose_2d, visibility, limb_delta_2d, limb_length_2d)

        # Apply limb transformer for inter-limb reasoning
        limb_features = self.transformer(limb_features)

        # Predict POF vectors
        pof = self.pof_head(limb_features)

        return pof

    def num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for checkpointing."""
        return {
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
        }


def create_pof_model(
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    verbose: bool = True,
) -> CameraPOFModel:
    """Create camera-space POF model with specified configuration.

    Args:
        d_model: Hidden dimension (default 128)
        num_heads: Number of attention heads (default 4)
        num_layers: Number of transformer layers (default 4)
        dim_feedforward: FFN dimension (default 256)
        dropout: Dropout rate (default 0.1)
        verbose: Print model info

    Returns:
        Initialized CameraPOFModel
    """
    model = CameraPOFModel(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )

    if verbose:
        print(f"Created CameraPOFModel:")
        print(f"  d_model={d_model}, heads={num_heads}, layers={num_layers}")
        print(f"  Parameters: {model.num_parameters():,}")

    return model


def load_pof_model(
    checkpoint_path: str,
    device: str = "cpu",
    verbose: bool = True,
) -> CameraPOFModel:
    """Load POF model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        verbose: Print loading info

    Returns:
        Loaded CameraPOFModel
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint or use defaults
    config = checkpoint.get("config", {})
    model = create_pof_model(
        d_model=config.get("d_model", 128),
        num_heads=config.get("num_heads", 4),
        num_layers=config.get("num_layers", 4),
        verbose=False,
    )

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    if verbose:
        print(f"Loaded CameraPOFModel from {checkpoint_path}")
        print(f"  Parameters: {model.num_parameters():,}")

    return model
