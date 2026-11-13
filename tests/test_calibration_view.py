from app.calibration_view import reliability_chart_spec


def test_reliability_chart_does_not_connect_observed_bins_or_rescale_axes():
    spec = reliability_chart_spec()
    reference, observed = spec["layer"]
    assert reference["mark"]["type"] == "line"
    assert observed["mark"]["type"] == "point"
    assert "shape" in observed["encoding"]
    for layer in spec["layer"]:
        assert layer["encoding"]["x"]["scale"]["domain"] == [0, 1]
        assert layer["encoding"]["y"]["scale"]["domain"] == [0, 1]
