# TinyTorchProfiler

TinyTorchProfiler is a lightweight Python library for profiling PyTorch models.

It helps users understand model performance beyond accuracy, including latency,
parameter count, model size, activation memory, layer-wise runtime, and likely
bottlenecks.

The goal of v0.1.0 is to stay small, readable, and practical for ML systems
workflows where model performance matters as much as model quality.

## Why This Library Matters

A model with good accuracy can still be too slow, too large, or too
memory-heavy for production use.

TinyTorchProfiler helps answer practical deployment questions:

- How many parameters does this model have?
- How many parameters are trainable?
- How large is the model in memory?
- What is the average forward-pass latency?
- Which layers are the slowest?
- Which layers create the largest activations?
- Does this model look suitable for deployment constraints?

## Installation

Install the published package from PyPI:

```bash
pip install tinytorchprofiler==0.1.2
```

From the project root, install the package in editable mode:

```bash
pip install -e .
```

For development, install the optional test dependencies:

```bash
pip install -e ".[dev]"
```

You can also install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quickstart

```python
from torch import nn

from tinytorchprofiler import profile_model


model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)

report = profile_model(
    model,
    input_shape=(1, 128),
    device="cpu",
    warmup=5,
    runs=20,
)

report.summary()
report.to_csv("profile.csv")
```

Show package metadata:

```python
import tinytorchprofiler as tinyprofiler

tinyprofiler.show()
```

Check whether the model fits a deployment budget:

```python
budget = report.check_budget(
    max_latency_ms=30,
    max_model_size_mb=25,
    max_parameters=5_000_000,
)

print(budget["passed"])
print(budget["checks"])
```

Score the model for a built-in deployment target:

```python
score = report.deployment_score("edge_cpu")

print(score["score"])
print(score["bottlenecks"])
```

Use strict custom deployment budgets:

```python
score = report.deployment_score(
    target="custom",
    max_latency_ms=0.1,
    max_model_size_mb=0.001,
)

print(score["passed"])
print(score["score"])
```

## Core API

```python
from tinytorchprofiler import profile_model

report = profile_model(
    model,
    input_shape=(1, 3, 224, 224),
    device="cpu",
    warmup=5,
    runs=20,
)

report.summary()
report.to_dict()
report.to_csv("profile.csv")
```

## Supported Metrics

TinyTorchProfiler v0.1.2 supports:

- Total parameter count
- Trainable parameter count
- Estimated model size in MB
- Average forward-pass latency in milliseconds
- CPU profiling
- CUDA profiling when available
- Layer-wise names
- Layer-wise module types
- Layer-wise input and output shapes
- Layer-wise parameter counts
- Approximate activation size in MB
- Approximate layer-wise forward latency
- Deployment budget checks
- Deployment readiness score for built-in and custom targets

## Deployment Readiness

TinyTorchProfiler can check model metrics against production-style deployment
budgets:

```python
result = report.check_budget(
    max_latency_ms=30,
    max_model_size_mb=25,
    max_parameters=5_000_000,
    max_activation_size_mb=64,
)
```

It can also calculate a simple 0-100 deployment readiness score:

```python
score = report.deployment_score("edge_cpu")
```

You can override any built-in target budget:

```python
score = report.deployment_score(
    "edge_cpu",
    max_latency_ms=10,
)
```

You can also use a fully custom target:

```python
score = report.deployment_score(
    target="custom",
    max_latency_ms=0.1,
    max_model_size_mb=0.001,
    max_parameters=1_000,
    max_activation_size_mb=1,
)
```

Built-in targets:

- `edge_cpu`
- `mobile`
- `server_cpu`
- `realtime_webcam`
- `custom`

The score is a lightweight heuristic based on latency, model size, parameter
count, and activation memory. It is intended as a fast first-pass signal, not a
replacement for production benchmarking on real hardware.

## Device Support

CPU profiling is supported by default:

```python
report = profile_model(model, input_shape=(1, 3, 224, 224), device="cpu")
```

CUDA profiling is supported when PyTorch detects an available CUDA device:

```python
report = profile_model(model, input_shape=(1, 3, 224, 224), device="cuda")
```

For CUDA timing, TinyTorchProfiler synchronizes the device around measured
regions with `torch.cuda.synchronize()`.

## Examples

Profile a small CNN:

```bash
python examples/01_profile_simple_cnn.py
```

This prints a summary and writes:

```text
simple_cnn_profile.csv
```

Profile a torchvision model if torchvision is installed:

```bash
python examples/02_profile_torchvision_model.py
```

If torchvision is missing, the example exits gracefully with a short message.

## Testing

Run the test suite with:

```bash
pytest
```

The v0.1.0 tests cover:

- Parameter counting
- Trainable parameter counting
- Model size estimation
- Tensor size estimation
- Report serialization with `to_dict()`
- Basic end-to-end profiling on a tiny model

## Notes on Profiling

TinyTorchProfiler uses PyTorch forward passes and hooks.

Layer-wise latency is approximate. It is useful for identifying likely
bottlenecks, but exact timing can vary depending on hardware, backend libraries,
CPU load, CUDA synchronization, and model structure.

Activation memory is estimated from forward outputs. It should be treated as a
helpful approximation, not as a replacement for full memory tracing.

## Roadmap

Planned future improvements:

- Batch-size scaling analysis
- Memory peak tracking
- Visualization
- ONNX export profiling
- ViT and DINOv2 examples

## Version

Current version: `0.1.2`
