"""Validated elemental-composition splits for the paired SuperCon CSV files."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# unique_m.csv has one amount column for each element from H through Rn. Keeping
# the schema explicit prevents material, critical_temp, or future metadata from
# accidentally becoming part of the composition key.
ELEMENT_COLUMNS = tuple(
    """
    H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni
    Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe
    Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg
    Tl Pb Bi Po At Rn
    """.split()
)
COMPOSITION_DECIMALS = 12


@dataclass(frozen=True)
class SuperconSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_groups: pd.Series
    validation_groups: pd.Series
    test_groups: pd.Series


def canonical_composition_groups(
    compositions: pd.DataFrame,
    decimals: int = COMPOSITION_DECIMALS,
) -> pd.Series:
    """Return deterministic keys for elemental ratios, independent of scale.

    Element columns are put in atomic-number order and each row is divided by
    its total elemental amount. The tracked source amounts have at most five
    decimal places; rounding normalized fractions to 12 decimal places removes
    floating-point noise while retaining substantially finer distinctions than
    the source data. Rounded zeros are omitted from the key consistently.
    """
    element_columns = [
        column for column in ELEMENT_COLUMNS if column in compositions
    ]
    if not element_columns:
        raise ValueError("Composition data contains no recognized element columns")
    if any(
        not pd.api.types.is_numeric_dtype(compositions[column])
        for column in element_columns
    ):
        raise ValueError("Elemental composition columns must be numeric")

    amounts = compositions.loc[:, element_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(amounts).all():
        raise ValueError(
            "Elemental composition columns contain missing or non-finite values"
        )
    if (amounts < 0).any():
        raise ValueError("Elemental composition amounts must be non-negative")

    totals = amounts.sum(axis=1)
    if (totals <= 0).any():
        raise ValueError("Every row must contain a positive elemental amount")

    normalized = np.round(amounts / totals[:, np.newaxis], decimals=decimals)
    keys = [
        tuple(
            (element, float(fraction))
            for element, fraction in zip(element_columns, row)
            if fraction != 0.0
        )
        for row in normalized
    ]
    return pd.Series(
        keys,
        index=compositions.index,
        name="canonical_composition",
        dtype=object,
    )


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
    compositions = pd.read_csv(material_path)
    if target not in data or target not in compositions:
        raise ValueError(f"Paired files must both contain the target column {target!r}")
    missing_elements = [
        column for column in ELEMENT_COLUMNS if column not in compositions.columns
    ]
    if missing_elements:
        raise ValueError(
            "Paired composition file is missing elemental columns: "
            + ", ".join(missing_elements)
        )
    if len(data) != len(compositions):
        raise ValueError("SuperCon files do not have matching row counts")
    if not np.allclose(
        data[target].to_numpy(), compositions[target].to_numpy(), equal_nan=True
    ):
        raise ValueError("SuperCon target values do not align row-for-row")
    return data, canonical_composition_groups(compositions)


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
        raise AssertionError("Canonical composition groups overlap across splits")

    return SuperconSplits(
        train=data.iloc[train_idx].reset_index(drop=True),
        validation=data.iloc[validation_idx].reset_index(drop=True),
        test=data.iloc[test_idx].reset_index(drop=True),
        train_groups=groups.iloc[train_idx].reset_index(drop=True),
        validation_groups=groups.iloc[validation_idx].reset_index(drop=True),
        test_groups=groups.iloc[test_idx].reset_index(drop=True),
    )
