"""Shared neural network layers for GNN models.

Contains reusable Graph Neural Network building blocks used by both
POF and joint refinement models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """Graph Convolutional layer with residual connection.

    Uses normalized adjacency (D^-1 A) for message passing.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual and (in_dim == out_dim)

        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass through GCN layer.

        Args:
            x: (batch, num_nodes, in_dim) node features
            adj: (num_nodes, num_nodes) adjacency matrix

        Returns:
            (batch, num_nodes, out_dim) updated node features
        """
        # Normalize adjacency: D^-1 A (row normalization)
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        adj_norm = adj / deg

        # Message passing: aggregate neighbor features
        h = torch.matmul(adj_norm, x)  # (batch, num_nodes, in_dim)

        # Transform
        h = self.linear(h)
        h = self.norm(h)
        h = F.gelu(h)
        h = self.dropout(h)

        # Residual connection
        if self.use_residual:
            h = h + x

        return h


class GATLayer(nn.Module):
    """Graph Attention layer (simplified version).

    Learns attention weights for neighbors instead of using fixed adjacency.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0

        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)

        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass through GAT layer.

        Args:
            x: (batch, num_nodes, in_dim)
            adj: (num_nodes, num_nodes) - used as attention mask

        Returns:
            (batch, num_nodes, out_dim)
        """
        batch_size, num_nodes, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)

        # (batch, heads, nodes, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Mask non-neighbors (where adj == 0)
        mask = (adj == 0).unsqueeze(0).unsqueeze(0)  # (1, 1, nodes, nodes)
        scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Aggregate
        out = torch.matmul(attn, v)  # (batch, heads, nodes, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, -1)
        out = self.out_proj(out)

        # Residual + norm
        out = self.norm(out + x)

        return out
