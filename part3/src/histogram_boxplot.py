#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PART3_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PART3_DIR / "data" / "train.csv"
OUTPUT_DIR = PART3_DIR / "outputs" / "target_distribution"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the SuperCon target distribution and report IQR outliers."
    )
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--target", default="critical_temp")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--prefix", default="critical_temp")
    parser.add_argument("--show", action="store_true", help="Display plots interactively.")
    return parser.parse_args()


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str,
    output_dir: Path,
    save_prefix: str,
    show: bool = False,
) -> pd.Series:
    """Save target histogram/boxplot figures and return IQR outliers."""
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    output_dir.mkdir(parents=True, exist_ok=True)
    y = df[target_col]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(y, bins=40, kde=True, edgecolor="black", ax=ax)
    ax.set_title(f"Histogram of {target_col}")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3)
    histogram_path = output_dir / f"{save_prefix}_hist.png"
    fig.savefig(histogram_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(x=y, orient="h", showfliers=True, ax=ax)
    mean_val = y.mean()
    ax.axvline(
        mean_val,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_val:.2f}",
    )
    ax.set_title(f"Boxplot of {target_col} with mean")
    ax.set_xlabel(target_col)
    ax.grid(alpha=0.3)
    boxplot_path = output_dir / f"{save_prefix}_boxplot.png"
    fig.savefig(boxplot_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    q1 = y.quantile(0.25)
    q3 = y.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = y[(y < lower) | (y > upper)]

    print("===== Outlier Summary =====")
    print(f"N outliers: {len(outliers)}")
    print(f"Lower bound: {lower:.3f}")
    print(f"Upper bound: {upper:.3f}")
    print(f"Min value:   {y.min():.3f}")
    print(f"Max value:   {y.max():.3f}")
    print(f"Saved histogram: {histogram_path}")
    print(f"Saved boxplot: {boxplot_path}")
    return outliers


def main() -> None:
    args = parse_args()
    if not args.data_path.is_file():
        raise FileNotFoundError(f"SuperCon dataset not found: {args.data_path}")
    df = pd.read_csv(args.data_path)
    plot_target_distribution(
        df,
        args.target,
        output_dir=args.output_dir,
        save_prefix=args.prefix,
        show=args.show,
    )


if __name__ == "__main__":
    main()
