"""GPU configuration for Pose2Sim ONNX Runtime acceleration.

This module monkey-patches Pose2Sim to use GPU execution providers
for LSTM inference, providing 3-10x speedup on augmentation.
"""

import onnxruntime as ort


def patch_pose2sim_gpu():
    """Patch Pose2Sim to use GPU for ONNX Runtime inference.

    This monkey-patches the InferenceSession creation in Pose2Sim's
    markerAugmentation module to explicitly use CUDA providers.

    Call this before importing or using Pose2Sim augmentation functions.
    """
    try:
        # Check if GPU providers are available
        available_providers = ort.get_available_providers()

        if 'CUDAExecutionProvider' not in available_providers:
            print("[GPU] Warning: CUDA provider not available, using CPU")
            return

        # Monkey-patch onnxruntime.InferenceSession
        original_init = ort.InferenceSession.__init__

        def patched_init(self, path_or_bytes, sess_options=None, providers=None, **kwargs):
            """Patched InferenceSession to use GPU by default."""
            if providers is None:
                # Use GPU providers if available
                providers = [
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }),
                    'CPUExecutionProvider',  # Fallback
                ]
                print(f"[GPU] Using CUDA provider for ONNX model: {path_or_bytes if isinstance(path_or_bytes, str) else 'model'}")

            original_init(self, path_or_bytes, sess_options, providers, **kwargs)

        ort.InferenceSession.__init__ = patched_init
        print("[GPU] Pose2Sim GPU acceleration enabled (CUDA)")

    except Exception as e:
        print(f"[GPU] Warning: Failed to patch Pose2Sim for GPU: {e}")


def get_gpu_info():
    """Get information about available GPU execution providers.

    Returns:
        dict: GPU provider information
    """
    providers = ort.get_available_providers()

    return {
        'available_providers': providers,
        'cuda_available': 'CUDAExecutionProvider' in providers,
        'tensorrt_available': 'TensorrtExecutionProvider' in providers,
    }
