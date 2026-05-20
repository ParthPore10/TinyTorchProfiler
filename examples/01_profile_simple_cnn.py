"""Profile a small CNN with TinyTorchProfiler."""

import torch
from torch import nn

from tinytorchprofiler import profile_model


class SimpleCNN(nn.Module):
    """Small convolutional network for profiling examples."""

    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


if __name__ == "__main__":
    model = SimpleCNN()

    report = profile_model(
        model,
        input_shape=(1, 3, 224, 224),
        device="cpu",
        warmup=5,
        runs=20,
    )

    report.summary()

    budget = report.check_budget(
        max_latency_ms=0.1,
        max_model_size_mb=0.001,
    )

    score = report.deployment_score(
        target="custom",
        max_latency_ms=0.1,
        max_model_size_mb=0.001,
    )

    print("\nBudget passed:", budget["passed"])

    print("\nDeployment score:")
    print(f"Target: {score['target']}")
    print(f"Score: {score['score']}/100")
    print(f"Passed: {score['passed']}")

    print("\nTop bottlenecks:")
    for layer in score["bottlenecks"]:
        print(
            f"- {layer['name']} ({layer['layer_type']}): "
            f"{layer['latency_ms']:.3f} ms, "
            f"{layer['activation_size_mb']:.2f} MB activation"
        )

    report.to_csv("simple_cnn_profile.csv")
    print("Saved layer profile to simple_cnn_profile.csv")
