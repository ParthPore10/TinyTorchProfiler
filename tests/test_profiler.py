from torch import nn

from tinytorchprofiler import profile_model


def test_profiler_runs_on_tiny_model() -> None:
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )

    report = profile_model(
        model,
        input_shape=(1, 4),
        device="cpu",
        warmup=1,
        runs=2,
    )

    assert report.total_parameters == 58
    assert report.trainable_parameters == 58
    assert report.model_size_mb > 0
    assert report.average_latency_ms > 0
    assert report.device == "cpu"
    assert report.input_shape == (1, 4)
    assert len(report.layer_profiles) == 3