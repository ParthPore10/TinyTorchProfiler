import torch
from torch import nn

from tinytorchprofiler.memory import (
    count_parameters,
    count_trainable_parameters,
    estimate_model_size_mb,
    estimate_tensor_size_mb,
)


def test_count_parameters() -> None:
    model = nn.Linear(4, 2)

    assert count_parameters(model) == 10


def test_count_trainable_parameters() -> None:
    model = nn.Linear(4, 2)
    model.bias.requires_grad = False

    assert count_trainable_parameters(model) == 8


def test_estimate_model_size_mb() -> None:
    model = nn.Linear(4, 2)

    expected_bytes = 10 * 4
    expected_mb = expected_bytes / (1024 * 1024)

    assert estimate_model_size_mb(model) == expected_mb


def test_estimate_tensor_size_mb() -> None:
    tensor = torch.zeros(2, 3, dtype=torch.float32)

    expected_mb = 6 * 4 / (1024 * 1024)

    assert estimate_tensor_size_mb(tensor) == expected_mb