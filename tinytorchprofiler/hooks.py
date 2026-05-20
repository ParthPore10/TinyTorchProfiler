"""Forward hook utilities for layer-wise profiling."""

import time
from typing import List

import torch
from torch import nn

from tinytorchprofiler.memory import count_layer_parameters, estimate_activation_size_mb
from tinytorchprofiler.report import LayerProfile
from tinytorchprofiler.utils import shape_of, synchronize_device


class LayerProfiler:
    """Collect layer-wise profiling information using PyTorch forward hooks."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.layer_profiles: List[LayerProfile] = []
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def register(self, model: nn.Module) -> None:
        """Register hooks on leaf modules of a model."""
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an instance of torch.nn.Module")

        for name, module in model.named_modules():
            if name == "":
                continue

            if len(list(module.children())) > 0:
                continue

            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def clear(self) -> None:
        """Clear collected layer profiles."""
        self.layer_profiles.clear()

    def remove(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()

        self._handles.clear()

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs, output) -> None:
            input_value = inputs[0] if inputs else None

            self.layer_profiles.append(
                LayerProfile(
                    name=name,
                    layer_type=module.__class__.__name__,
                    input_shape=shape_of(input_value),
                    output_shape=shape_of(output),
                    parameters=count_layer_parameters(module),
                    activation_size_mb=estimate_activation_size_mb(output),
                    latency_ms=0.0,
                )
            )

        return hook


def measure_layer_latencies(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: torch.device,
) -> List[LayerProfile]:
    """Measure approximate per-layer forward latency using pre/post hooks."""
    profiles: List[LayerProfile] = []
    handles: List[torch.utils.hooks.RemovableHandle] = []
    start_times = {}

    def should_profile(module: nn.Module) -> bool:
        return len(list(module.children())) == 0

    def make_pre_hook(name: str):
        def pre_hook(module: nn.Module, inputs) -> None:
            synchronize_device(device)
            start_times[name] = time.perf_counter()

        return pre_hook

    def make_post_hook(name: str):
        def post_hook(module: nn.Module, inputs, output) -> None:
            synchronize_device(device)
            elapsed_ms = (time.perf_counter() - start_times[name]) * 1000.0
            input_value = inputs[0] if inputs else None

            profiles.append(
                LayerProfile(
                    name=name,
                    layer_type=module.__class__.__name__,
                    input_shape=shape_of(input_value),
                    output_shape=shape_of(output),
                    parameters=count_layer_parameters(module),
                    activation_size_mb=estimate_activation_size_mb(output),
                    latency_ms=elapsed_ms,
                )
            )

        return post_hook

    for name, module in model.named_modules():
        if name == "":
            continue

        if not should_profile(module):
            continue

        handles.append(module.register_forward_pre_hook(make_pre_hook(name)))
        handles.append(module.register_forward_hook(make_post_hook(name)))

    try:
        model.eval()
        with torch.no_grad():
            _ = model(sample_input)
    finally:
        for handle in handles:
            handle.remove()

    return profiles