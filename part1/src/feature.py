import argparse
from pathlib import Path

import numpy as np
import scipy.io

PART1_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PART1_DIR / "data" / "MNISTmini.mat"


def print_feature_shapes(data):
    print(f"train_fea1 shape: {data['train_fea1'].shape}")
    print(f"train_gnd1 shape: {data['train_gnd1'].shape}")
    print(f"test_fea1 shape: {data['test_fea1'].shape}")
    print(f"test_gnd1 shape: {data['test_gnd1'].shape}")
    print(" ")

def print_label_counts(data):
    print(f"Number of training instances with label 5: {np.sum(data['train_gnd1'] == 5)}")
    print(f"Number of training instances with label 6: {np.sum(data['train_gnd1'] == 6)}")
    print(f"Number of testing instances with label 5: {np.sum(data['test_gnd1'] == 5)}")
    print(f"Number of testing instances with label 6: {np.sum(data['test_gnd1'] == 6)}")
    print("")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the Part 1 MNIST MAT file.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_path.is_file():
        raise FileNotFoundError(f"MNISTmini dataset not found: {args.data_path}")
    data = scipy.io.loadmat(args.data_path)
    required = {"train_fea1", "train_gnd1", "test_fea1", "test_gnd1"}
    missing = required.difference(data)
    if missing:
        raise KeyError(f"Missing MAT keys: {sorted(missing)}")
    print_feature_shapes(data)
    print_label_counts(data)


if __name__ == "__main__":
    main()
