"""Main profiling API for TinyTorchProfiler."""

import time
from typing import Tuple

import torch
from torch import nn

from tinytorchprofiler.hooks import measure_layer_latencies
from tinytorchprofiler.memory import (
    count_parameters,
    count_trainable_parameters,
    estimate_model_size_mb,
)
from tinytorchprofiler.report import ProfileReport
from tinytorchprofiler.utils import (
    resolve_device,
    synchronize_device,
    validate_input_shape,
    validate_iterations,
)


def profile_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cpu",
    warmup: int = 5,
    runs: int = 20,
) -> ProfileReport:
    """Profile a PyTorch model and return a structured report.

    Args:
        model: PyTorch model to profile.
        input_shape: Shape of the synthetic input tensor, including batch size.
        device: Device to profile on. Supported values are "cpu" and "cuda".
        warmup: Number of warmup forward passes before timing.
        runs: Number of measured forward passes.

    Returns:
        A ProfileReport containing model-level and layer-level metrics.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")

    validate_input_shape(input_shape)
    validate_iterations(warmup, runs)

    torch_device = resolve_device(device)
    model = model.to(torch_device)
    model.eval()

    sample_input = torch.randn(*input_shape, device=torch_device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_input)

        synchronize_device(torch_device)

        start_time = time.perf_counter()

        for _ in range(runs):
            _ = model(sample_input)

        synchronize_device(torch_device)

        average_latency_ms = ((time.perf_counter() - start_time) / runs) * 1000.0

    layer_profiles = measure_layer_latencies(model, sample_input, torch_device)

    return ProfileReport(
        total_parameters=count_parameters(model),
        trainable_parameters=count_trainable_parameters(model),
        model_size_mb=estimate_model_size_mb(model),
        average_latency_ms=average_latency_ms,
        device=str(torch_device),
        input_shape=input_shape,
        layer_profiles=layer_profiles,
    )