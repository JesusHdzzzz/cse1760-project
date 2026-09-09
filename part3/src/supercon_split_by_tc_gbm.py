#!/usr/bin/env python3

"""Target-conditioned diagnostic analysis; not an end-to-end predictor."""

import argparse
import json
from pathlib import Path

import h2o
import pandas as pd
from h2o.estimators import H2OGradientBoostingEstimator

from h2o_utils import init_h2o, save_h2o_model, shutdown_h2o_if_owned
from supercon_utils import load_supercon_pair, material_group_split

PART3_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PART3_DIR / "data" / "train.csv"
MATERIAL_PATH = PART3_DIR / "data" / "unique_m.csv"
OUTPUT_DIR = PART3_DIR / "outputs" / "supercon_target_conditioned_diagnostic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run target-conditioned SuperCon bucket diagnostics. This analysis "
            "requires the true target for routing and is not deployable."
        )
    )
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--material-data-path", type=Path, default=MATERIAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--target", default="critical_temp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument(
        "--bucket-mode", choices=["quantiles", "thresholds"], default="quantiles"
    )
    parser.add_argument("--q-low", type=float, default=0.3333)
    parser.add_argument("--q-high", type=float, default=0.6667)
    parser.add_argument("--tc-low-max", type=float)
    parser.add_argument("--tc-high-min", type=float)
    parser.add_argument("--ntrees", type=int, default=800)
    parser.add_argument("--max-depth", type=int, default=11)
    parser.add_argument("--learn-rate", type=float, default=0.03)
    parser.add_argument("--sample-rate", type=float, default=0.8)
    parser.add_argument("--col-sample-rate", type=float, default=0.8)
    parser.add_argument("--min-rows", type=int, default=10)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--stopping-rounds", type=int, default=5)
    parser.add_argument("--stopping-tolerance", type=float, default=0.001)
    parser.add_argument("--stopping-metric", default="RMSE")
    parser.add_argument("--max-mem-size", default="8G")
    parser.add_argument("--keep-h2o-cluster", action="store_true")
    return parser.parse_args()


def bucket_bounds(training: pd.DataFrame, target: str, args):
    if args.bucket_mode == "quantiles":
        if not 0 < args.q_low < args.q_high < 1:
            raise ValueError("Require 0 < --q-low < --q-high < 1")
        return (
            float(training[target].quantile(args.q_low)),
            float(training[target].quantile(args.q_high)),
        )
    if args.tc_low_max is None or args.tc_high_min is None:
        raise ValueError("Threshold mode requires --tc-low-max and --tc-high-min")
    if args.tc_low_max >= args.tc_high_min:
        raise ValueError("--tc-low-max must be less than --tc-high-min")
    return args.tc_low_max, args.tc_high_min


def make_buckets(df: pd.DataFrame, target: str, bounds):
    low, high = bounds
    return {
        "low": df[df[target] <= low].copy(),
        "medium": df[(df[target] > low) & (df[target] < high)].copy(),
        "high": df[df[target] >= high].copy(),
    }


def to_h2o_frame(df: pd.DataFrame, target: str):
    frame = h2o.H2OFrame(df)
    frame[target] = frame[target].asnumeric()
    return frame


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
    bounds = bucket_bounds(splits.train, args.target, args)
    train_buckets = make_buckets(splits.train, args.target, bounds)
    validation_buckets = make_buckets(splits.validation, args.target, bounds)
    test_buckets = make_buckets(splits.test, args.target, bounds)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("WARNING: true critical_temp routes every row; metrics are diagnostic only.")
    print(f"Bucket boundaries learned from training targets: {bounds}")
    owned_cluster = init_h2o(args.max_mem_size)
    try:
        rows = []
        saved_models = {}
        for name in ("low", "medium", "high"):
            train = to_h2o_frame(train_buckets[name], args.target)
            validation = to_h2o_frame(validation_buckets[name], args.target)
            test = to_h2o_frame(test_buckets[name], args.target)
            features = [column for column in train.columns if column != args.target]
            model_kwargs = dict(
                ntrees=args.ntrees,
                max_depth=args.max_depth,
                learn_rate=args.learn_rate,
                sample_rate=args.sample_rate,
                col_sample_rate=args.col_sample_rate,
                min_rows=args.min_rows,
                seed=args.seed,
            )
            if args.early_stopping:
                model_kwargs.update(
                    stopping_rounds=args.stopping_rounds,
                    stopping_tolerance=args.stopping_tolerance,
                    stopping_metric=args.stopping_metric,
                )
            model = H2OGradientBoostingEstimator(**model_kwargs)
            model.train(
                x=features,
                y=args.target,
                training_frame=train,
                validation_frame=validation,
            )
            valid_perf = model.model_performance(validation)
            test_perf = model.model_performance(test)
            rows.append(
                {
                    "bucket": name,
                    "train_rows": train.nrows,
                    "validation_rows": validation.nrows,
                    "test_rows": test.nrows,
                    "validation_rmse": valid_perf.rmse(),
                    "validation_mae": valid_perf.mae(),
                    "diagnostic_test_rmse": test_perf.rmse(),
                    "diagnostic_test_mae": test_perf.mae(),
                    "target_routed": True,
                    "deployable": False,
                }
            )
            model_path = save_h2o_model(
                model,
                args.output_dir,
                f"supercon_target_conditioned_{name}_model",
            )
            saved_models[name] = model_path.name
            print(f"Saved {name} diagnostic model: {model_path}")

        summary = pd.DataFrame(rows)
        summary.to_csv(args.output_dir / "target_conditioned_bucket_metrics.csv", index=False)
        metadata = {
            "status": "target-conditioned diagnostic; not deployable",
            "seed": args.seed,
            "split_strategy": "normalized elemental-composition group split",
            "bucket_mode": args.bucket_mode,
            "bucket_bounds_learned_from_training_targets": list(bounds),
            "model_params": {
                "ntrees": args.ntrees,
                "max_depth": args.max_depth,
                "learn_rate": args.learn_rate,
                "sample_rate": args.sample_rate,
                "col_sample_rate": args.col_sample_rate,
                "min_rows": args.min_rows,
                "early_stopping": args.early_stopping,
                "stopping_rounds": args.stopping_rounds,
                "stopping_tolerance": args.stopping_tolerance,
                "stopping_metric": args.stopping_metric,
            },
            "saved_model_files": saved_models,
            "rows": {
                "train": len(splits.train),
                "validation": len(splits.validation),
                "test": len(splits.test),
            },
        }
        (args.output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        (args.output_dir / "README.txt").write_text(
            "These models are target-conditioned diagnostics. The true critical_temp "
            "selects a bucket, so their metrics are not end-to-end predictive performance.\n"
            f"Boundaries learned from the training split: {bounds}\n"
        )
        print(summary.to_string(index=False))
    finally:
        shutdown_h2o_if_owned(owned_cluster, args.keep_h2o_cluster)


if __name__ == "__main__":
    main()
