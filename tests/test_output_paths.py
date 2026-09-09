from pathlib import Path

import pandas as pd
import pytest

import h2o_utils
from histogram_boxplot import plot_target_distribution
from xgb_experiment import prepare_output_dir


def test_h2o_model_save_uses_absolute_selected_directory_and_filename(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "selected-output"
    output_dir.mkdir()
    calls = []

    def fake_save_model(model, path, filename, force):
        calls.append((model, path, filename, force))
        return str(Path(path) / filename)

    monkeypatch.setattr(h2o_utils.h2o, "save_model", fake_save_model)
    model = object()
    saved_path = h2o_utils.save_h2o_model(model, output_dir, "named_model")

    assert saved_path == output_dir.resolve() / "named_model"
    assert calls == [(model, str(output_dir.resolve()), "named_model", True)]


@pytest.mark.parametrize("filename", ["", ".", "..", "../escape", "/tmp/escape"])
def test_h2o_model_save_rejects_path_like_filenames(tmp_path, filename):
    with pytest.raises(ValueError):
        h2o_utils.save_h2o_model(object(), tmp_path, filename)


def test_h2o_model_save_rejects_unexpected_returned_path(tmp_path, monkeypatch):
    output_dir = tmp_path / "selected-output"
    output_dir.mkdir()
    monkeypatch.setattr(
        h2o_utils.h2o,
        "save_model",
        lambda *args, **kwargs: str(tmp_path / "unexpected-model"),
    )

    with pytest.raises(RuntimeError, match="outside its expected path"):
        h2o_utils.save_h2o_model(object(), output_dir, "named_model")


def test_histogram_prefix_cannot_escape_output_directory(tmp_path):
    data = pd.DataFrame({"critical_temp": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError):
        plot_target_distribution(
            data,
            "critical_temp",
            tmp_path / "selected-output",
            "../escape",
        )

    assert not (tmp_path / "escape_hist.png").exists()
    assert not (tmp_path / "escape_boxplot.png").exists()


def test_histogram_writes_only_to_selected_output_directory(tmp_path):
    data = pd.DataFrame({"critical_temp": [1.0, 2.0, 3.0, 4.0]})
    output_dir = tmp_path / "selected-output"

    plot_target_distribution(
        data,
        "critical_temp",
        output_dir,
        "tiny_distribution",
    )

    files = {
        path.relative_to(tmp_path)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert files == {
        Path("selected-output/tiny_distribution_hist.png"),
        Path("selected-output/tiny_distribution_boxplot.png"),
    }


def test_part2_output_directory_guard(tmp_path):
    output_dir = tmp_path / "part2-run"
    prepare_output_dir(output_dir, overwrite=False)
    assert output_dir.is_dir()

    (output_dir / "existing.txt").write_text("existing\n")
    with pytest.raises(FileExistsError):
        prepare_output_dir(output_dir, overwrite=False)
