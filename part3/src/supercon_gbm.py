#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

import h2o
import pandas as pd
from h2o.estimators import H2OGradientBoostingEstimator

from h2o_utils import init_h2o, shutdown_h2o_if_owned
from supercon_utils import load_supercon_pair, material_group_split

PART3_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PART3_DIR / "data" / "train.csv"
MATERIAL_PATH = PART3_DIR / "data" / "unique_m.csv"
OUTPUT_DIR = PART3_DIR / "outputs" / "supercon_gbm"

BASE_CONFIG = {
    "name": "gbm_base",
    "ntrees": 500,
    "max_depth": 8,
    "learn_rate": 0.05,
    "sample_rate": 0.8,
    "col_sample_rate": 0.8,
    "min_rows": 5,
}
GBM_SWEEP_CONFIGS = [
    {
        "name": "gbm_n1100_d11_lr03",
        "ntrees": 1100,
        "max_depth": 11,
        "learn_rate": 0.03,
        "sample_rate": 0.8,
        "col_sample_rate": 0.8,
        "min_rows": 10,
    },
    {
        "name": "gbm_n1100_d10_lr03",
        "ntrees": 1100,
        "max_depth": 10,
        "learn_rate": 0.03,
        "sample_rate": 0.8,
        "col_sample_rate": 0.8,
        "min_rows": 10,
    },
    {
        "name": "gbm_n1100_d12_lr02",
        "ntrees": 1100,
        "max_depth": 12,
        "learn_rate": 0.02,
        "sample_rate": 0.8,
        "col_sample_rate": 0.8,
        "min_rows": 10,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and evaluate H2O GBMs for SuperCon regression."
    )
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--material-data-path", type=Path, default=MATERIAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--target", default="critical_temp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--iqr-multiplier", type=float, default=1.5)
    parser.add_argument("--no-outlier-filter", action="store_true")
    parser.add_argument("--run-gbm-sweep", action="store_true")
    parser.add_argument("--max-mem-size", default="8G")
    parser.add_argument("--keep-h2o-cluster", action="store_true")
    return parser.parse_args()


def training_outlier_bounds(y: pd.Series, multiplier: float):
    q1 = float(y.quantile(0.25))
    q3 = float(y.quantile(0.75))
    iqr = q3 - q1
    return q1 - multiplier * iqr, q3 + multiplier * iqr


def filter_by_bounds(df: pd.DataFrame, target: str, bounds):
    lower, upper = bounds
    return df[df[target].between(lower, upper)].reset_index(drop=True)


def to_h2o_frame(df: pd.DataFrame, target: str):
    frame = h2o.H2OFrame(df)
    frame[target] = frame[target].asnumeric()
    return frame


def build_model(config: dict, seed: int) -> H2OGradientBoostingEstimator:
    return H2OGradientBoostingEstimator(
        ntrees=config["ntrees"],
        max_depth=config["max_depth"],
        learn_rate=config["learn_rate"],
        sample_rate=config["sample_rate"],
        col_sample_rate=config["col_sample_rate"],
        min_rows=config["min_rows"],
        seed=seed,
        stopping_rounds=0,
        score_tree_interval=5,
    )


def validation_result(model, validation, config: dict, elapsed: float):
    performance = model.model_performance(validation)
    return {
        **config,
        "validation_rmse": float(performance.rmse()),
        "validation_mae": float(performance.mae()),
        "train_time_seconds": elapsed,
    }


def main() -> None:
    args = parse_args()
    data, groups = load_supercon_pair(
        args.data_path, args.material_data_path, args.target
    )
    splits = material_group_split(
        data,
        groups,
        validation_size=args.validation_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    bounds = None
    training = splits.train
    if not args.no_outlier_filter:
        bounds = training_outlier_bounds(training[args.target], args.iqr_multiplier)
        training = filter_by_bounds(training, args.target, bounds)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    owned_cluster = init_h2o(args.max_mem_size)
    try:
        train_h2o = to_h2o_frame(training, args.target)
        validation_h2o = to_h2o_frame(splits.validation, args.target)
        features = [column for column in train_h2o.columns if column != args.target]

        configs = [BASE_CONFIG]
        if args.run_gbm_sweep:
            configs += GBM_SWEEP_CONFIGS
        candidate_rows = []
        for config in configs:
            print(f"Training selection candidate: {config['name']}")
            start = time.perf_counter()
            model = build_model(config, args.seed)
            model.train(
                x=features,
                y=args.target,
                training_frame=train_h2o,
                validation_frame=validation_h2o,
            )
            candidate_rows.append(
                validation_result(
                    model, validation_h2o, config, time.perf_counter() - start
                )
            )

        selection = pd.DataFrame(candidate_rows).sort_values("validation_rmse")
        selection.to_csv(args.output_dir / "model_selection.csv", index=False)
        selected_name = selection.iloc[0]["name"]
        selected_config = next(config for config in configs if config["name"] == selected_name)

        final_training = pd.concat(
            [splits.train, splits.validation], ignore_index=True
        )
        if bounds is not None:
            final_training = filter_by_bounds(final_training, args.target, bounds)
        final_train_h2o = to_h2o_frame(final_training, args.target)
        final_model = build_model(selected_config, args.seed)
        final_model.train(
            x=features, y=args.target, training_frame=final_train_h2o
        )

        # The unfiltered test split is evaluated once after candidate selection.
        test_h2o = to_h2o_frame(splits.test, args.target)
        test_performance = final_model.model_performance(test_h2o)
        model_path = h2o.save_model(
            final_model, path=str(args.output_dir), force=True
        )
        metadata = {
            "status": "current after material-aware split and validation-only selection",
            "seed": args.seed,
            "target": args.target,
            "split_strategy": "group shuffle split using row-aligned unique_m.csv material formulas",
            "rows": {
                "train_before_filter": len(splits.train),
                "train_after_filter": len(training),
                "validation": len(splits.validation),
                "test": len(splits.test),
                "final_training_after_filter": len(final_training),
            },
            "unique_materials": {
                "train": splits.train_groups.nunique(),
                "validation": splits.validation_groups.nunique(),
                "test": splits.test_groups.nunique(),
            },
            "outlier_policy": "bounds learned from initial training target; filtering applied only to model-fitting rows",
            "training_outlier_bounds": list(bounds) if bounds else None,
            "selection_metric": "validation RMSE",
            "selected_config": selected_config,
            "final_test_rmse": float(test_performance.rmse()),
            "final_test_mae": float(test_performance.mae()),
            "saved_model": model_path,
        }
        (args.output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        print(f"Selected configuration: {selected_name}")
        print(f"Final test RMSE: {metadata['final_test_rmse']:.4f}")
        print(f"Final test MAE: {metadata['final_test_mae']:.4f}")
    finally:
        shutdown_h2o_if_owned(owned_cluster, args.keep_h2o_cluster)


if __name__ == "__main__":
    main()
