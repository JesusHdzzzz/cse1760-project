import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "part1/src/feature.py",
    "part1/src/logistic_regression.py",
    "part1/src/random_forest.py",
    "part2/src/xgb_mnist_pixels.py",
    "part2/src/xgb_mnist_lenet.py",
    "part3/src/stroke_random_forest.py",
    "part3/src/stroke_h2o_gbm.py",
    "part3/src/histogram_boxplot.py",
    "part3/src/supercon_gbm.py",
    "part3/src/supercon_split_by_tc_gbm.py",
]


@pytest.mark.parametrize("script", SCRIPTS)
def test_cli_help_does_not_run_experiment(script):
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_all_executable_scripts_have_main_guards():
    for script in SCRIPTS:
        text = (ROOT / script).read_text()
        assert "def main(" in text
        assert 'if __name__ == "__main__":' in text
