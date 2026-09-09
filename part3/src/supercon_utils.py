"""Validated material-aware splits for the paired SuperCon CSV files."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


@dataclass(frozen=True)
class SuperconSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_groups: pd.Series
    validation_groups: pd.Series
    test_groups: pd.Series


def load_supercon_pair(
    data_path: Path,
    material_path: Path,
    target: str = "critical_temp",
):
    data_path = Path(data_path)
    material_path = Path(material_path)
    for path in (data_path, material_path):
        if not path.is_file():
            raise FileNotFoundError(f"SuperCon data file not found: {path}")

    data = pd.read_csv(data_path)
    materials = pd.read_csv(material_path)
    if target not in data or target not in materials or "material" not in materials:
        raise ValueError("Paired files must contain critical_temp and a material column")
    if len(data) != len(materials):
        raise ValueError("SuperCon files do not have matching row counts")
    if not np.allclose(
        data[target].to_numpy(), materials[target].to_numpy(), equal_nan=True
    ):
        raise ValueError("SuperCon target values do not align row-for-row")
    if materials["material"].isna().any():
        raise ValueError("Material grouping column contains missing values")
    return data, materials["material"].astype(str)


def material_group_split(
    data: pd.DataFrame,
    groups: pd.Series,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> SuperconSplits:
    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("Validation and test sizes must be positive and sum to less than 1")

    holdout_size = validation_size + test_size
    first = GroupShuffleSplit(n_splits=1, test_size=holdout_size, random_state=seed)
    train_idx, holdout_idx = next(first.split(data, groups=groups))
    holdout = data.iloc[holdout_idx]
    holdout_groups = groups.iloc[holdout_idx]
    relative_test_size = test_size / holdout_size
    second = GroupShuffleSplit(
        n_splits=1, test_size=relative_test_size, random_state=seed + 1
    )
    validation_rel, test_rel = next(
        second.split(holdout, groups=holdout_groups)
    )
    validation_idx = holdout_idx[validation_rel]
    test_idx = holdout_idx[test_rel]

    split_groups = [
        set(groups.iloc[index].tolist())
        for index in (train_idx, validation_idx, test_idx)
    ]
    if any(
        split_groups[left].intersection(split_groups[right])
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise AssertionError("Material groups overlap across splits")

    return SuperconSplits(
        train=data.iloc[train_idx].reset_index(drop=True),
        validation=data.iloc[validation_idx].reset_index(drop=True),
        test=data.iloc[test_idx].reset_index(drop=True),
        train_groups=groups.iloc[train_idx].reset_index(drop=True),
        validation_groups=groups.iloc[validation_idx].reset_index(drop=True),
        test_groups=groups.iloc[test_idx].reset_index(drop=True),
    )
