#!/usr/bin/env python3
import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
import h2o
from h2o.estimators import H2OGradientBoostingEstimator


@dataclass
class Metrics:
    n_rows: int
    tc_min: float
    tc_max: float
    tc_mean: float
    train_rmse: float
    valid_rmse: float
    test_rmse: float
    train_mae: float
    valid_mae: float
    test_mae: float


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default= "../Part3/images/superconductivty-data/train.csv", required=False, help="Path to SuperCon CSV")
    ap.add_argument("--target", default="critical_temp", help="Target column (Tc)")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--bucket_mode",
        choices=["quantiles", "thresholds"],
        default="quantiles",
        help="How to define Low/Med/High Tc buckets",
    )
    ap.add_argument(
        "--q_low",
        type=float,
        default=0.3333,
        help="Quantile cutoff between Low and Medium (quantiles mode)",
    )
    ap.add_argument(
        "--q_high",
        type=float,
        default=0.6667,
        help="Quantile cutoff between Medium and High (quantiles mode)",
    )
    ap.add_argument(
        "--tc_low_max",
        type=float,
        default=None,
        help="Max Tc for Low bucket (thresholds mode)",
    )
    ap.add_argument(
        "--tc_high_min",
        type=float,
        default=None,
        help="Min Tc for High bucket (thresholds mode)",
    )

    # proportion of data for test/valid sets
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--valid_frac", type=float, default=0.15)

    # GBM hyperparameters, same for all buckets for reusability
    ap.add_argument("--ntrees", type=int, default=800)
    ap.add_argument("--max_depth", type=int, default=11)
    ap.add_argument("--learn_rate", type=float, default=0.03)
    ap.add_argument("--sample_rate", type=float, default=0.8)
    ap.add_argument("--col_sample_rate", type=float, default=0.8)
    ap.add_argument("--min_rows", type=int, default=10)

    ap.add_argument("--outdir", default="tc_bucket_results")
    return ap.parse_args()


def make_buckets(df: pd.DataFrame, target: str, args) -> Dict[str, pd.DataFrame]:
    y = df[target]

    if args.bucket_mode == "quantiles":
        q_low = float(y.quantile(args.q_low))
        q_high = float(y.quantile(args.q_high))

        low = df[df[target] <= q_low].copy()
        med = df[(df[target] > q_low) & (df[target] < q_high)].copy()
        high = df[df[target] >= q_high].copy()

        print(f"[INFO] Quantile thresholds:")
        print(f"       Low <= {q_low:.3f} K")
        print(f"       Med  ( {q_low:.3f}, {q_high:.3f} ) K")
        print(f"       High >= {q_high:.3f} K")

        return {"low": low, "medium": med, "high": high}

    # thresholds mode
    if args.tc_low_max is None or args.tc_high_min is None:
        raise ValueError(
            "thresholds mode requires --tc_low_max and --tc_high_min"
        )
    if args.tc_low_max >= args.tc_high_min:
        raise ValueError("--tc_low_max must be < --tc_high_min")

    low = df[df[target] <= args.tc_low_max].copy()
    med = df[(df[target] > args.tc_low_max) & (df[target] < args.tc_high_min)].copy()
    high = df[df[target] >= args.tc_high_min].copy()

    print(f"[INFO] Manual thresholds:")
    print(f"       Low <= {args.tc_low_max:.3f} K")
    print(f"       Med  ( {args.tc_low_max:.3f}, {args.tc_high_min:.3f} ) K")
    print(f"       High >= {args.tc_high_min:.3f} K")

    return {"low": low, "medium": med, "high": high}


def train_eval_gbm(
    h2o_frame,
    target: str,
    args,
) -> Tuple[H2OGradientBoostingEstimator, Metrics]:
    # Split within bucket: train / valid / test
    # Use ratios: first split off test, then split remaining into train/valid.
    train_valid, test = h2o_frame.split_frame(
        ratios=[1.0 - args.test_frac],
        seed=args.seed
    )
    # Now split train_valid into train/valid
    # valid_frac is with respect to whole dataset; convert to fraction of remaining
    remain = 1.0 - args.test_frac
    valid_ratio_of_remain = args.valid_frac / remain
    train, valid = train_valid.split_frame(
        ratios=[1.0 - valid_ratio_of_remain],
        seed=args.seed
    )

    x = [c for c in h2o_frame.columns if c != target]
    y = target

    gbm_kwargs = dict(
        ntrees=args.ntrees,
        max_depth=args.max_depth,
        learn_rate=args.learn_rate,
        sample_rate=args.sample_rate,
        col_sample_rate=args.col_sample_rate,
        min_rows=args.min_rows,
        seed=args.seed,
    )

    if args.early_stopping:
        gbm_kwargs.update(
            stopping_rounds=args.stopping_rounds,
            stopping_tolerance=args.stopping_tolerance,
            stopping_metric=args.stopping_metric,
        )

    model = H2OGradientBoostingEstimator(**gbm_kwargs)
    model.train(x=x, y=y, training_frame=train, validation_frame=valid)

    # Metrics
    perf_train = model.model_performance(train=True)
    perf_valid = model.model_performance(valid=True)
    perf_test = model.model_performance(test)

    # Tc stats (from the full bucket frame)
    tc_series = h2o_frame[y].as_data_frame(use_multi_thread=True)[y]

    m = Metrics(
        n_rows=h2o_frame.nrows,
        tc_min=float(tc_series.min()),
        tc_max=float(tc_series.max()),
        tc_mean=float(tc_series.mean()),
        train_rmse=float(perf_train.rmse()),
        valid_rmse=float(perf_valid.rmse()),
        test_rmse=float(perf_test.rmse()),
        train_mae=float(perf_train.mae()),
        valid_mae=float(perf_valid.mae()),
        test_mae=float(perf_test.mae()),
    )
    return model, m


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found in columns")

    # Create buckets in pandas first and then convert to H2OFrames
    buckets = make_buckets(df, args.target, args)

    h2o.init()

    rows = []
    for name, bucket_df in buckets.items():
        if len(bucket_df) < 100:
            print(f"[WARN] Bucket '{name}' only has {len(bucket_df)} rows — metrics may be noisy.")

        hf = h2o.H2OFrame(bucket_df)

        # Ensure target is numeric
        # If it's imported as enum then forced conversion
        hf[args.target] = hf[args.target].asnumeric()

        print(f"\n[INFO] Training bucket '{name}' (n={hf.nrows}, Tc range {bucket_df[args.target].min():.3f}..{bucket_df[args.target].max():.3f})")
        model, metrics = train_eval_gbm(hf, args.target, args)

        model_path = h2o.save_model(model=model, path=args.outdir, force=True)
        print(f"[INFO] Saved model: {model_path}")

        rows.append({
            "bucket": name,
            "n_rows": metrics.n_rows,
            "tc_min": metrics.tc_min,
            "tc_mean": metrics.tc_mean,
            "tc_max": metrics.tc_max,
            "train_RMSE": metrics.train_rmse,
            "valid_RMSE": metrics.valid_rmse,
            "test_RMSE": metrics.test_rmse,
            "train_MAE": metrics.train_mae,
            "valid_MAE": metrics.valid_mae,
            "test_MAE": metrics.test_mae,
        })

    summary = pd.DataFrame(rows).sort_values("bucket")
    print("\n===== Summary (GBM by Tc bucket) =====")
    print(summary.to_string(index=False))

    out_csv = os.path.join(args.outdir, "gbm_tc_bucket_summary.csv")
    summary.to_csv(out_csv, index=False)
    print(f"\n[INFO] Saved summary CSV: {out_csv}")

    h2o.shutdown(prompt=False)


if __name__ == "__main__":
    main()