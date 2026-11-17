"""Loader for the locally built official 2DMamba CUDA extension."""

import torch  # Load libtorch symbols before importing the pybind11 extension.

try:
    from .pscan import bwd, fwd
except (ImportError, OSError) as exc:
    raise RuntimeError(
        "2DMamba CUDA kernel is not built for this Python/PyTorch/CUDA environment"
    ) from exc

__all__ = ["fwd", "bwd"]
