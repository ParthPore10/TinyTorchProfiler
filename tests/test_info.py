import tinytorchprofiler


def test_show_prints_package_info(capsys) -> None:
    tinytorchprofiler.show()

    output = capsys.readouterr().out

    assert "TinyTorchProfiler" in output
    assert "Version:" in output
    assert "Author: Parth Pore" in output
