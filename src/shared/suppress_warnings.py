"""Warning suppression utilities.

Suppresses common noisy warnings from TensorFlow, ONNX Runtime, and
other dependencies that don't affect functionality.
"""

import os
import warnings
import logging


def suppress_common_warnings():
    """Suppress common noisy warnings from dependencies.

    Call this early in scripts to reduce console noise from:
    - TensorFlow logging and GPU detection
    - ONNX Runtime provider warnings
    - NumPy deprecation warnings
    - MediaPipe logging
    """
    # TensorFlow: Suppress verbose logging
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    # Suppress TensorFlow CUDA warnings when using CPU
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", ""))

    # ONNX Runtime: Suppress provider warnings
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)

    # Suppress Python warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Suppress specific messages
    warnings.filterwarnings(
        "ignore",
        message=".*CUDA.*not available.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*TensorRT.*",
    )
