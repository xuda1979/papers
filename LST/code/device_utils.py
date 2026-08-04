"""Device selection utility supporting Apple MPS, CUDA, and Ascend NPU.

This module provides a single `get_device()` function that picks the best
available accelerator.  When `torch_npu` is installed and an Ascend NPU is
visible, it returns an NPU device; otherwise it falls back to CUDA, then MPS,
then CPU.

Usage (replace the existing device line in each experiment script):

    from device_utils import get_device
    device = get_device()
    print(f"Using device: {device}", flush=True)

NPU notes
---------
On an Ascend NPU host with `torch_npu` installed, the device string is "npu:0"
(or "npu" for the default).  `torch.npu.empty_cache()` is the analogue of
`torch.mps.empty_cache()` / `torch.cuda.empty_cache()`.
"""
from __future__ import annotations
import torch


def get_device() -> torch.device:
    """Return the best available compute device.

    Priority: npu > cuda > mps > cpu.
    """
    # Ascend NPU via torch_npu
    try:
        import torch_npu  # noqa: F401  (registers the "npu" backend)
        if torch.npu.is_available():
            return torch.device("npu:0")
    except ImportError:
        pass

    # NVIDIA CUDA
    if torch.cuda.is_available():
        return torch.device("cuda:0")

    # Apple Silicon MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def empty_cache(device: torch.device) -> None:
    """Empty the accelerator cache for the given device (no-op on CPU)."""
    if device.type == "npu":
        try:
            torch.npu.empty_cache()
        except Exception:
            pass
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    d = get_device()
    print(f"Selected device: {d}")
