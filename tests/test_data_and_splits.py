from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_utils import split_data
from stroke_utils import impute_bmi_from_training
from stroke_random_forest import build_pipeline
from supercon_utils import load_supercon_pair, material_group_split
from utils_mnist import convert_mnist_labels, load_mnist_mat
from xgb_experiment import build_pipeline as build_xgb_pipeline

ROOT = Path(__file__).resolve().parents[1]


def test_course_mnist_label_mapping_preserves_digit_identity():
    raw = np.array([1, 2, 5, 9, 10])
    assert convert_mnist_labels(raw).tolist() == [1, 2, 5, 9, 0]


def test_already_converted_mnist_labels_are_unchanged():
    labels = np.arange(10)
    assert np.array_equal(convert_mnist_labels(labels), labels)


def test_invalid_mnist_labels_are_rejected():
    with pytest.raises(ValueError):
        convert_mnist_labels(np.array([0, 11]))


def test_part1_split_is_disjoint_and_deterministic():
    X = np.arange(600).reshape(300, 2)
    y = np.tile([0, 1], 150)
    first = split_data(X, y, 100, 80, 60, random_state=7)
    second = split_data(X, y, 100, 80, 60, random_state=7)
    for left, right in zip(first, second):
        assert np.array_equal(left, right)
    row_sets = [set(array[:, 0]) for array in first[:3]]
    assert row_sets[0].isdisjoint(row_sets[1])
    assert row_sets[0].isdisjoint(row_sets[2])
    assert row_sets[1].isdisjoint(row_sets[2])


def test_bmi_imputation_uses_training_median_only():
    train = pd.DataFrame({"bmi": [10.0, 20.0, np.nan]})
    test = pd.DataFrame({"bmi": [np.nan, 1000.0]})
    train_result, test_result, median = impute_bmi_from_training(train, test)
    assert median == 15.0
    assert train_result["bmi"].tolist() == [10.0, 20.0, 15.0]
    assert test_result["bmi"].tolist() == [15.0, 1000.0]


def test_random_forest_pipeline_fits_imputer_on_training_rows():
    X_train = pd.DataFrame(
        {
            "age": np.arange(10, dtype=float),
            "bmi": [10.0, 20.0, 30.0, 40.0, np.nan] * 2,
            "gender": ["Female", "Male"] * 5,
        }
    )
    y_train = pd.Series([0, 1] * 5)
    model = build_pipeline(X_train, seed=2)
    model.set_params(classifier__n_estimators=2).fit(X_train, y_train)
    numeric_imputer = model.named_steps["preprocess"].named_transformers_["numeric"]
    assert numeric_imputer.statistics_[1] == 25.0


def test_xgboost_pipeline_smoke_fit():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(100, 12)).astype(np.float32)
    y = np.tile(np.arange(10), 10)
    model = build_xgb_pipeline(seed=4).set_params(
        pca__n_components=4,
        model__n_estimators=1,
        model__max_depth=2,
        model__n_jobs=1,
    )
    model.fit(X, y)
    assert model.predict(X[:5]).shape == (5,)


def test_material_group_split_has_no_group_overlap():
    groups = pd.Series(np.repeat([f"m{i}" for i in range(30)], 2))
    data = pd.DataFrame({"feature": np.arange(60), "critical_temp": np.arange(60)})
    splits = material_group_split(data, groups, seed=3)
    group_sets = [
        set(splits.train_groups),
        set(splits.validation_groups),
        set(splits.test_groups),
    ]
    assert group_sets[0].isdisjoint(group_sets[1])
    assert group_sets[0].isdisjoint(group_sets[2])
    assert group_sets[1].isdisjoint(group_sets[2])
    assert len(splits.train) + len(splits.validation) + len(splits.test) == len(data)


@pytest.mark.skipif(
    not (ROOT / "part2" / "data" / "MNIST.mat").is_file(),
    reason="course MAT file is not present",
)
def test_reviewed_mnist_mat_schema_and_mapping():
    path = ROOT / "part2" / "data" / "MNIST.mat"
    X, y = load_mnist_mat(path, "train_fea", "train_gnd")
    assert X.shape == (60000, 784)
    assert set(np.unique(y)) == set(range(10))
    assert int((y == 0).sum()) == 5923
    assert int((y == 1).sum()) == 6742


def test_tracked_supercon_files_are_row_aligned():
    data, groups = load_supercon_pair(
        ROOT / "part3" / "data" / "train.csv",
        ROOT / "part3" / "data" / "unique_m.csv",
    )
    assert data.shape == (21263, 82)
    assert len(groups) == len(data)
    assert groups.nunique() == 15542
