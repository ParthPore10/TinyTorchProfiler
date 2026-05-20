from typing import Any, Optional, Tuple

import torch


def resolve_device(device: str) -> torch.device:
    """Resolve and validate the requested profiling device."""
    if not isinstance(device, str):
        raise TypeError("device must be a string")

    normalized = device.lower()

    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")

    if normalized == "cpu":
        return torch.device("cpu")

    raise ValueError("device must be either 'cpu' or 'cuda'")


def synchronize_device(device: torch.device) -> None:
    """Synchronize CUDA work when profiling on GPU."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def shape_of(value: Any) -> Any:
    """Return a lightweight shape description for tensors or nested outputs."""
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)

    if isinstance(value, dict):
        return {key: shape_of(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [shape_of(item) for item in value]

    return None


def validate_input_shape(input_shape: Tuple[int, ...]) -> None:
    """Validate an input shape tuple."""
    if not isinstance(input_shape, tuple):
        raise TypeError("input_shape must be a tuple of positive integers")

    if not input_shape:
        raise ValueError("input_shape cannot be empty")

    if not all(isinstance(dim, int) and dim > 0 for dim in input_shape):
        raise ValueError("input_shape must contain only positive integers")


def validate_iterations(warmup: int, runs: int) -> None:
    """Validate warmup and measured iteration counts."""
    if not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")

    if not isinstance(runs, int) or runs <= 0:
        raise ValueError("runs must be a positive integer")