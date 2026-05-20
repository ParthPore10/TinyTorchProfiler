"""Package metadata helpers."""

__version__ = "0.1.1"
__author__ = "Parth Pore"
__project_name__ = "TinyTorchProfiler"


def show() -> None:
    """Print basic package information."""
    print(__project_name__)
    print("-" * len(__project_name__))
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print("Description: Lightweight PyTorch profiling for deployment readiness.")
    print("Main API: profile_model(model, input_shape, device='cpu')")
