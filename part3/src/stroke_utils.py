"""Data preparation helpers shared by the Part 3 stroke experiments."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_stroke_data(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Stroke dataset not found: {path}")
    df = pd.read_csv(path)
    required = {"id", "gender", "bmi", "stroke"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing stroke columns: {sorted(missing)}")
    return df[df["gender"] != "Other"].copy()


def stratified_stroke_split(df: pd.DataFrame, test_size: float, seed: int):
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["stroke"],
    )
    return train.copy(), test.copy()


def impute_bmi_from_training(train: pd.DataFrame, test: pd.DataFrame):
    median = float(train["bmi"].median())
    if pd.isna(median):
        raise ValueError("Training split has no observed BMI values")
    train = train.copy()
    test = test.copy()
    train["bmi"] = train["bmi"].fillna(median)
    test["bmi"] = test["bmi"].fillna(median)
    return train, test, median
