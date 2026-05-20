from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


DEPLOYMENT_TARGETS: Dict[str, Dict[str, float]] = {
    "edge_cpu": {
        "max_latency_ms": 30.0,
        "max_model_size_mb": 25.0,
        "max_parameters": 5_000_000,
        "max_activation_size_mb": 64.0,
    },
    "mobile": {
        "max_latency_ms": 50.0,
        "max_model_size_mb": 50.0,
        "max_parameters": 10_000_000,
        "max_activation_size_mb": 128.0,
    },
    "server_cpu": {
        "max_latency_ms": 100.0,
        "max_model_size_mb": 250.0,
        "max_parameters": 100_000_000,
        "max_activation_size_mb": 512.0,
    },
    "realtime_webcam": {
        "max_latency_ms": 33.0,
        "max_model_size_mb": 100.0,
        "max_parameters": 25_000_000,
        "max_activation_size_mb": 256.0,
    },
}


@dataclass
class LayerProfile:
    name: str
    layer_type: str
    input_shape: Any
    output_shape: Any
    parameters: int
    activation_size_mb: float
    latency_ms: float


@dataclass
class ProfileReport:
    total_parameters: int
    trainable_parameters: int
    model_size_mb: float
    average_latency_ms: float
    device: str
    input_shape: Sequence[int]
    layer_profiles: List[LayerProfile]

    def summary(self) -> None:
        print("TinyTorchProfiler Report")
        print("-" * 32)
        print(f"Device: {self.device}")
        print(f"Input shape: {tuple(self.input_shape)}")
        print(f"Total parameters: {self.total_parameters:,}")
        print(f"Trainable parameters: {self.trainable_parameters:,}")
        print(f"Model size: {self.model_size_mb:.2f} MB")
        print(f"Average latency: {self.average_latency_ms:.3f} ms")
        print(f"Layers profiled: {len(self.layer_profiles)}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "model_size_mb": self.model_size_mb,
            "average_latency_ms": self.average_latency_ms,
            "device": self.device,
            "input_shape": tuple(self.input_shape),
            "layer_profiles": [asdict(layer) for layer in self.layer_profiles],
        }

    def check_budget(
        self,
        max_latency_ms: Optional[float] = None,
        max_model_size_mb: Optional[float] = None,
        max_parameters: Optional[int] = None,
        max_activation_size_mb: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Check the report against deployment budgets.

        Returns a dictionary with an overall pass/fail flag plus per-metric
        results. Budgets set to None are ignored.
        """
        checks = []

        if max_latency_ms is not None:
            checks.append(
                {
                    "metric": "average_latency_ms",
                    "value": self.average_latency_ms,
                    "budget": max_latency_ms,
                    "passed": self.average_latency_ms <= max_latency_ms,
                }
            )

        if max_model_size_mb is not None:
            checks.append(
                {
                    "metric": "model_size_mb",
                    "value": self.model_size_mb,
                    "budget": max_model_size_mb,
                    "passed": self.model_size_mb <= max_model_size_mb,
                }
            )

        if max_parameters is not None:
            checks.append(
                {
                    "metric": "total_parameters",
                    "value": self.total_parameters,
                    "budget": max_parameters,
                    "passed": self.total_parameters <= max_parameters,
                }
            )

        if max_activation_size_mb is not None:
            max_activation = self.max_activation_size_mb
            checks.append(
                {
                    "metric": "max_activation_size_mb",
                    "value": max_activation,
                    "budget": max_activation_size_mb,
                    "passed": max_activation <= max_activation_size_mb,
                }
            )

        return {
            "passed": all(check["passed"] for check in checks),
            "checks": checks,
        }

    def deployment_score(
        self,
        target: str = "edge_cpu",
        max_latency_ms: Optional[float] = None,
        max_model_size_mb: Optional[float] = None,
        max_parameters: Optional[int] = None,
        max_activation_size_mb: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Score deployment readiness for a named target.

        The score is a lightweight 0-100 heuristic based on latency, model
        size, parameter count, and activation memory budgets.
        """
        custom_budgets = {
            "max_latency_ms": max_latency_ms,
            "max_model_size_mb": max_model_size_mb,
            "max_parameters": max_parameters,
            "max_activation_size_mb": max_activation_size_mb,
        }

        has_custom_budget = any(value is not None for value in custom_budgets.values())

        if target == "custom":
            if not has_custom_budget:
                raise ValueError("custom target requires at least one budget")
            budgets = custom_budgets
        elif target in DEPLOYMENT_TARGETS:
            budgets = {
                **DEPLOYMENT_TARGETS[target],
                **{
                    key: value
                    for key, value in custom_budgets.items()
                    if value is not None
                },
            }
        else:
            supported = ", ".join(sorted(DEPLOYMENT_TARGETS))
            raise ValueError(
                f"unknown target '{target}'. Supported targets: {supported}, custom"
            )

        budget_result = self.check_budget(
            max_latency_ms=budgets["max_latency_ms"],
            max_model_size_mb=budgets["max_model_size_mb"],
            max_parameters=(
                int(budgets["max_parameters"])
                if budgets["max_parameters"] is not None
                else None
            ),
            max_activation_size_mb=budgets["max_activation_size_mb"],
        )

        penalties = []
        for check in budget_result["checks"]:
            if check["value"] <= check["budget"]:
                penalties.append(0.0)
                continue

            over_budget_ratio = (check["value"] - check["budget"]) / check["budget"]
            penalties.append(min(25.0, over_budget_ratio * 25.0))

        score = max(0, round(100.0 - sum(penalties)))

        return {
            "target": target,
            "score": score,
            "passed": budget_result["passed"],
            "checks": budget_result["checks"],
            "bottlenecks": self.bottlenecks(),
        }

    def bottlenecks(self, top_k: int = 3) -> List[Dict[str, Any]]:
        """Return the slowest layers in the report."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        layers = sorted(
            self.layer_profiles,
            key=lambda layer: layer.latency_ms,
            reverse=True,
        )

        return [asdict(layer) for layer in layers[:top_k]]

    @property
    def max_activation_size_mb(self) -> float:
        """Return the largest recorded layer activation size in MB."""
        if not self.layer_profiles:
            return 0.0

        return max(layer.activation_size_mb for layer in self.layer_profiles)

    def to_csv(self, path: str) -> None:
        output_path = Path(path)
        rows = [asdict(layer) for layer in self.layer_profiles]

        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "CSV export requires pandas. Please install it."
            ) from exc
        pd.DataFrame(rows).to_csv(output_path, index=False)
