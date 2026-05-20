from tinytorchprofiler.report import LayerProfile, ProfileReport


def test_report_to_dict() -> None:
    layer = LayerProfile(
        name="linear",
        layer_type="Linear",
        input_shape=(1, 4),
        output_shape=(1, 2),
        parameters=10,
        activation_size_mb=0.001,
        latency_ms=0.1,
    )

    report = ProfileReport(
        total_parameters=10,
        trainable_parameters=10,
        model_size_mb=0.00004,
        average_latency_ms=0.2,
        device="cpu",
        input_shape=(1, 4),
        layer_profiles=[layer],
    )

    data = report.to_dict()

    assert data["total_parameters"] == 10
    assert data["trainable_parameters"] == 10
    assert data["device"] == "cpu"
    assert data["input_shape"] == (1, 4)
    assert data["layer_profiles"][0]["name"] == "linear"


def test_report_check_budget_passes() -> None:
    layer = LayerProfile(
        name="linear",
        layer_type="Linear",
        input_shape=(1, 4),
        output_shape=(1, 2),
        parameters=10,
        activation_size_mb=0.001,
        latency_ms=0.1,
    )

    report = ProfileReport(
        total_parameters=10,
        trainable_parameters=10,
        model_size_mb=0.00004,
        average_latency_ms=0.2,
        device="cpu",
        input_shape=(1, 4),
        layer_profiles=[layer],
    )

    result = report.check_budget(
        max_latency_ms=1.0,
        max_model_size_mb=1.0,
        max_parameters=100,
        max_activation_size_mb=1.0,
    )

    assert result["passed"] is True
    assert len(result["checks"]) == 4


def test_report_check_budget_fails() -> None:
    report = ProfileReport(
        total_parameters=1_000,
        trainable_parameters=1_000,
        model_size_mb=5.0,
        average_latency_ms=20.0,
        device="cpu",
        input_shape=(1, 4),
        layer_profiles=[],
    )

    result = report.check_budget(max_latency_ms=10.0, max_parameters=500)

    assert result["passed"] is False
    assert result["checks"][0]["metric"] == "average_latency_ms"
    assert result["checks"][0]["passed"] is False


def test_report_deployment_score() -> None:
    layer = LayerProfile(
        name="linear",
        layer_type="Linear",
        input_shape=(1, 4),
        output_shape=(1, 2),
        parameters=10,
        activation_size_mb=0.001,
        latency_ms=0.1,
    )

    report = ProfileReport(
        total_parameters=10,
        trainable_parameters=10,
        model_size_mb=0.00004,
        average_latency_ms=0.2,
        device="cpu",
        input_shape=(1, 4),
        layer_profiles=[layer],
    )

    result = report.deployment_score("edge_cpu")

    assert result["target"] == "edge_cpu"
    assert result["score"] == 100
    assert result["passed"] is True
    assert result["bottlenecks"][0]["name"] == "linear"


def test_report_custom_deployment_score_can_fail() -> None:
    report = ProfileReport(
        total_parameters=1_000,
        trainable_parameters=1_000,
        model_size_mb=5.0,
        average_latency_ms=20.0,
        device="cpu",
        input_shape=(1, 4),
        layer_profiles=[],
    )

    result = report.deployment_score(
        target="custom",
        max_latency_ms=10.0,
        max_model_size_mb=1.0,
    )

    assert result["target"] == "custom"
    assert result["score"] < 100
    assert result["passed"] is False


def test_report_deployment_score_allows_budget_overrides() -> None:
    report = ProfileReport(
        total_parameters=1_000,
        trainable_parameters=1_000,
        model_size_mb=5.0,
        average_latency_ms=20.0,
        device="cpu",
        input_shape=(1, 4),
        layer_profiles=[],
    )

    result = report.deployment_score("edge_cpu", max_latency_ms=10.0)

    assert result["target"] == "edge_cpu"
    assert result["passed"] is False
    assert result["checks"][0]["budget"] == 10.0
