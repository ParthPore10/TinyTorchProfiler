from typing import Any
import torch
from torch import nn

_bytes_per_mb = 1024*1024

def count_parameters(model:nn.Module)->int:
    """ this returns the number of parameters in a PyTorch model """
    if not isinstance(model, nn.Module):
        raise TypeError("Input must be a PyTorch module")
    return sum(parameters.numel() for parameters in model.parameters())

def count_trainable_parameters(model:nn.Module)->int:
    """this returns the number of trainable parameters in a PyTorch model """
    if not isinstance(model, nn.Module):
        raise TypeError("Input must be a PyTorch module")
    return sum(parameters.numel() for parameters in model.parameters() if parameters.requires_grad)

def estimate_model_size_mb(model:nn.Module)->float:
    """this returns the estimated size of the model in MB """
    if not isinstance(model, nn.Module):
        raise TypeError("Input must be a PyTorch module")
    total_bytes =0

    for parameter in model.parameters():
        total_bytes += parameter.numel() * parameter.element_size()
    return total_bytes / _bytes_per_mb

def estimate_tensor_size_mb(tensor:torch.Tensor)->float:
    """this returns the estimated size of the tensor in MB """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor")
    return tensor.numel() * tensor.element_size() / _bytes_per_mb

def estimate_activation_size_mb(output: Any) -> float:
    """this returns the estimated size of the activations in MB """
    if isinstance(output, torch.Tensor):
        return estimate_tensor_size_mb(output)

    if isinstance(output, dict):
        return sum(estimate_activation_size_mb(value) for value in output.values())

    if isinstance(output, (list, tuple)):
        return sum(estimate_activation_size_mb(item) for item in output)

    return 0.0

def count_layer_parameters(layer: nn.Module) -> int:
    """Return parameter count for a single layer, excluding child modules."""
    if not isinstance(layer, nn.Module):
        raise TypeError("layer must be an instance of torch.nn.Module")

    return sum(parameter.numel() for parameter in layer.parameters(recurse=False))
