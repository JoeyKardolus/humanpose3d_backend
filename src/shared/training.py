"""Shared training utilities.

Contains reusable training infrastructure used across training scripts.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler


def setup_device(device: str = "auto") -> str:
    """Setup compute device with automatic detection.

    Args:
        device: Device specification ("auto", "cuda", "cpu", "mps")

    Returns:
        Selected device string
    """
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device


def setup_amp(
    fp16: bool = False,
    bf16: bool = False,
) -> Tuple[bool, torch.dtype, GradScaler]:
    """Setup automatic mixed precision training.

    Args:
        fp16: Use FP16 mixed precision
        bf16: Use BF16 mixed precision (preferred on modern hardware)

    Returns:
        Tuple of (use_amp, amp_dtype, scaler)
    """
    use_amp = fp16 or bf16

    if bf16:
        amp_dtype = torch.bfloat16
    elif fp16:
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32

    # GradScaler only needed for FP16 (BF16 has sufficient dynamic range)
    scaler = GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    return use_amp, amp_dtype, scaler


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    best_val_loss: float,
    output_path: Path,
    config: Optional[Dict[str, Any]] = None,
    extra_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler (optional)
        epoch: Current epoch number
        best_val_loss: Best validation loss so far
        output_path: Output file path
        config: Model configuration dict
        extra_data: Additional data to save
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_val_loss": best_val_loss,
    }

    # Add model config (try model.get_config() if available)
    if config is not None:
        checkpoint["config"] = config
    elif hasattr(model, "get_config"):
        checkpoint["config"] = model.get_config()

    # Merge extra data
    if extra_data:
        checkpoint.update(extra_data)

    # Ensure parent directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, output_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load tensors to

    Returns:
        Checkpoint dict with metadata (epoch, best_val_loss, config, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
        "config": checkpoint.get("config", {}),
    }


class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False
