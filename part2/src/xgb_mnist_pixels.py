#!/usr/bin/env python3

import argparse
from pathlib import Path

from xgb_experiment import ExperimentConfig, run_experiment

PART2_DIR = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Part 2 XGBoost experiment on MNIST pixel features."
    )
    parser.add_argument(
        "--data-path", type=Path, default=PART2_DIR / "data" / "MNIST.mat"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PART2_DIR / "outputs" / "pixels"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=55000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        name="mnist_pixels_xgboost",
        display_name="XGBoost on MNIST pixel features",
        data_path=args.data_path,
        output_dir=args.output_dir,
        cv_folds=2,
        search_iterations=4,
        parameter_distributions={
            "model__max_depth": [4, 5, 6],
            "model__colsample_bytree": [0.5, 0.7],
            "model__subsample": [0.8, 0.9],
            "model__reg_lambda": [0.1, 1.0],
        },
    )
    run_experiment(config, args.seed, args.train_size, args.overwrite)


if __name__ == "__main__":
    main()
