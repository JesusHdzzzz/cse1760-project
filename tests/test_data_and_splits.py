from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_utils import split_data
from stroke_utils import impute_bmi_from_training
from stroke_random_forest import build_pipeline
from stroke_h2o_gbm import (
    BMI_MISSING_VALUE_POLICY,
    select_model_and_f2_threshold,
    split_raw_stroke_data,
)
from supercon_utils import (
    canonical_composition_groups,
    load_supercon_pair,
    material_group_split,
)
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


def test_h2o_outer_split_preserves_raw_missing_bmi_values():
    data = pd.DataFrame(
        {
            "id": np.arange(40),
            "bmi": [
                np.nan if index % 5 == 0 else 20.0 + index
                for index in range(40)
            ],
            "stroke": np.tile([0, 1], 20),
        }
    )

    train, test = split_raw_stroke_data(data, test_size=0.2, seed=42)

    assert train["bmi"].isna().sum() + test["bmi"].isna().sum() == 8
    assert set(train["id"]).isdisjoint(test["id"])
    assert BMI_MISSING_VALUE_POLICY == "native H2O GBM missing-value handling"


def test_h2o_model_and_threshold_selection_uses_training_oof_data_only():
    class FrameResult:
        def __init__(self, frame):
            self.frame = frame

        def as_data_frame(self):
            return self.frame

    class Model:
        def cross_validation_holdout_predictions(self):
            return FrameResult(pd.DataFrame({"p1": [0.1, 0.8, 0.2, 0.9]}))

    class SortedGrid:
        models = [Model()]

    class Grid:
        def get_grid(self, sort_by, decreasing):
            assert sort_by == "aucpr"
            assert decreasing is True
            return SortedGrid()

    train = {"stroke": FrameResult(pd.DataFrame({"stroke": [0, 1, 0, 1]}))}

    model, cv_f2, threshold = select_model_and_f2_threshold(
        Grid(), train, "stroke"
    )

    assert isinstance(model, Model)
    assert cv_f2 == 1.0
    assert 0.2 < threshold <= 0.8


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


def test_canonical_composition_groups_ignore_formula_and_column_order():
    compositions = pd.DataFrame(
        {
            "material": ["Rb1Eu1Fe4As4", "Eu1Rb1Fe4As4"],
            "Rb": [1.0, 1.0],
            "Eu": [1.0, 1.0],
            "As": [4.0, 4.0],
            "Fe": [4.0, 4.0],
            "critical_temp": [36.5, 36.0],
        }
    )
    reversed_columns = compositions.loc[:, list(reversed(compositions.columns))]

    groups = canonical_composition_groups(compositions)
    reordered_groups = canonical_composition_groups(reversed_columns)

    assert groups.iloc[0] == groups.iloc[1]
    assert groups.tolist() == reordered_groups.tolist()


def test_canonical_composition_groups_normalize_proportional_amounts():
    compositions = pd.DataFrame(
        {
            "B": [0.3, 3.0],
            "Ru": [0.7, 7.0],
            "material": ["B0.3Ru0.7", "B3Ru7"],
        }
    )

    groups = canonical_composition_groups(compositions)

    assert groups.iloc[0] == groups.iloc[1]


def test_canonical_composition_groups_remove_harmless_float_noise():
    compositions = pd.DataFrame(
        {
            "B": [0.3, 0.1 + 0.2],
            "Ru": [0.7, 0.7],
        }
    )

    groups = canonical_composition_groups(compositions)

    assert groups.iloc[0] == groups.iloc[1]


def test_canonical_composition_groups_keep_meaningful_differences():
    compositions = pd.DataFrame(
        {
            "B": [0.3, 0.300001],
            "Ru": [0.7, 0.699999],
        }
    )

    groups = canonical_composition_groups(compositions)

    assert groups.iloc[0] != groups.iloc[1]


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


def test_tracked_supercon_files_are_row_aligned_and_canonically_grouped():
    data, groups = load_supercon_pair(
        ROOT / "part3" / "data" / "train.csv",
        ROOT / "part3" / "data" / "unique_m.csv",
    )
    assert data.shape == (21263, 82)
    assert data.drop(columns="critical_temp").shape[1] == 81
    assert len(groups) == len(data)
    assert groups.nunique() == 15164

    compositions = pd.read_csv(ROOT / "part3" / "data" / "unique_m.csv")
    by_formula = pd.Series(groups.to_numpy(), index=compositions["material"])
    assert by_formula["Rb1Eu1Fe4As4"] == by_formula["Eu1Rb1Fe4As4"]
    assert by_formula["B0.3Ru0.7"] == by_formula["B3Ru7"].iloc[0]

    splits = material_group_split(data, groups, seed=42)
    group_sets = [
        set(splits.train_groups),
        set(splits.validation_groups),
        set(splits.test_groups),
    ]
    assert group_sets[0].isdisjoint(group_sets[1])
    assert group_sets[0].isdisjoint(group_sets[2])
    assert group_sets[1].isdisjoint(group_sets[2])
    assert len(splits.train) + len(splits.validation) + len(splits.test) == len(data)

    feature_sets = [
        set(
            split.drop(columns="critical_temp").itertuples(index=False, name=None)
        )
        for split in (splits.train, splits.validation, splits.test)
    ]
    assert feature_sets[0].isdisjoint(feature_sets[1])
    assert feature_sets[0].isdisjoint(feature_sets[2])
    assert feature_sets[1].isdisjoint(feature_sets[2])
