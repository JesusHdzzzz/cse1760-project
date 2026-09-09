# CSE 176 Machine Learning Project

## Project Overview

This repository contains three stages of undergraduate machine-learning coursework,
later reorganized into reproducible command-line experiments. The work progresses
from classical binary classification on MNIST, through multiclass XGBoost models,
to imbalanced classification and materials-property regression with scikit-learn
and H2O.

The code emphasizes defensible train/validation/test responsibilities and records
run metadata. Full experiments have not been rerun since the latest correctness
changes, so historical metrics are not presented as current results.

## Repository Structure

- `part1/`: logistic regression and random forest classification of MNIST digits 5 and 6.
- `part2/`: XGBoost classification of all MNIST digits using pixels or precomputed LeNet features.
- `part3/`: stroke classification and SuperCon critical-temperature regression/diagnostics.
- `tests/`: lightweight loader, label, split, preprocessing, and CLI smoke tests.
- `ARTIFACTS.md`: status and provenance of reports, plots, and result files.

## Part 1

Part 1 uses `MNISTmini.mat` and compares a scaled logistic-regression pipeline
with a random forest. A deterministic stratified 1,000/1,000/1,000
train/validation/test subset is used by default. Hyperparameters are selected by
five-fold cross-validation on the training split; the selected model is refit on
train plus validation and evaluated once on test. The current scripts are ready
to rerun. Tracked plots and PDFs are historical coursework artifacts.

## Part 2

Part 2 compares XGBoost on 784 raw pixel features and 800 externally supplied
LeNet features. Scaling, 80%-variance PCA, and XGBoost are contained in one
scikit-learn pipeline, including within cross-validation. Search parameters are
selected by stratified CV, tree count by a held-out validation split, and test is
used once after final refitting. The pixel experiment retains two-fold CV because
of its coursework-scale runtime; this gives a noisier estimate than the
three-fold LeNet search.

The MAT encoding is converted semantically: raw label `10` becomes digit `0`,
while raw labels `1` through `9` remain unchanged. All tracked Part 2 metrics,
plots, reports, and locally saved models predate this correction and are stale
until both experiments are rerun.

## Part 3

The stroke experiments compare a class-weighted scikit-learn random forest and
an H2O GBM. Both use stratified holdouts and training-only preprocessing. The
random forest selects by cross-validation average precision; the H2O grid selects
by H2O's cross-validation AUCPR. Test metrics are calculated only after model and
threshold selection.

The SuperCon GBM uses normalized elemental-composition groups from the
row-aligned `unique_m.csv` file to prevent reordered or proportionally scaled
versions of the same composition appearing across train, validation, and test.
Optional target-outlier bounds are learned from training targets and applied
only to model-fitting rows. Candidate GBMs are selected by validation RMSE,
followed by one final evaluation on the unfiltered test split.
The separate target-bucket script is explicitly diagnostic: it uses the true
critical temperature to route rows and is not a deployable predictor.

## Setup

The smoke-tested environment used Python 3.13.11 and the package versions in
`requirements.txt`. Python 3.10 or newer is recommended; only the listed Python
3.13 environment has been checked in this revision.

```bash
./setup.sh
source .venv/bin/activate
```

Java is not needed for Parts 1 or 2. A Java runtime compatible with the pinned
H2O version is required only for the H2O-based Part 3 scripts. Local cluster
startup was smoke-tested with this environment's OpenJDK 25.0.1; check H2O's
support matrix before selecting Java for another environment.

## Data

Part 1 and Part 2 MAT files are intentionally ignored by Git and must be placed
manually at the paths below. They were supplied for the course; the repository
does not contain an authoritative upstream URL, retrieval date, license, or
LeNet checkpoint, so their provenance is incomplete.

| Path | Expected shape/keys | SHA-256 of reviewed local file |
| --- | --- | --- |
| `part1/data/MNISTmini.mat` | train 60,000 x 100; test 10,000 x 100; `*_fea1`, `*_gnd1` | `c33d5d8782df9f080d9efd1bb9823a58c87d53686c2970f5e70aa3c9c3db05ba` |
| `part2/data/MNIST.mat` | train 60,000 x 784; test 10,000 x 784; `train_fea`, `train_gnd`, `test_fea`, `test_gnd` | `b22e20253d929c06f2a925f0e90596a908c2b8cb73346428eec57364d6ecfb2f` |
| `part2/data/MNIST-LeNet5.mat` | train 60,000 x 800; test 10,000 x 800; same keys | `a1ad9e80de3a473a451ac6c3d32b7c1111b83fd9450e648616cfafe5436674a2` |

Part 3 data is tracked. The stroke file matches Federico Soriano's
[Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).
The paired SuperCon files match the [UCI Superconductivity Data set](https://archive.ics.uci.edu/dataset/464/superconductivty%2Bdata)
(DOI `10.24432/C53P47`, CC BY 4.0). Checksums and schemas are in
`part3/README.md`.

## Running Experiments

Commands below work from the repository root. Use `--help` to inspect options.
Generated artifacts go under each part's ignored `outputs/` directory.

```bash
python part1/src/feature.py
python part1/src/logistic_regression.py
python part1/src/random_forest.py

python part2/src/xgb_mnist_pixels.py --output-dir part2/outputs/pixels-corrected-labels
python part2/src/xgb_mnist_lenet.py --output-dir part2/outputs/lenet-corrected-labels

python part3/src/stroke_random_forest.py
python part3/src/stroke_h2o_gbm.py
python part3/src/histogram_boxplot.py
python part3/src/supercon_gbm.py --run-gbm-sweep
python part3/src/supercon_split_by_tc_gbm.py
```

Part 2 refuses to write into a nonempty output directory unless `--overwrite` is
given explicitly. Use new output directories for portfolio reruns.

## Results

No quantitative result is currently claimed for the revised implementation.
Part 1 figures, the Part 2 reports/results, and Part 3 course materials were
produced by earlier code or methodology and are retained only as historical
coursework artifacts. After rerunning, current scripts write model-selection
tables, final metrics, fitted models, figures, seeds, splits, and configuration
metadata beneath `part*/outputs/`.

## Reproducibility

- Repository-relative defaults avoid dependence on the current working directory.
- Every executable has a `main()` guard and supports `--help` without loading data.
- Important data paths, output directories, split sizes, CV folds, and seeds are CLI options.
- Preprocessing is fit only on training data or inside cross-validation pipelines.
- Output metadata records the selected configuration and final evaluation.
- Reviewed data shapes and SHA-256 values are documented; loaders validate keys and alignment.

## Known Limitations

- Course-supplied MNIST and LeNet feature provenance is incomplete, and the LeNet feature extractor/checkpoint is unavailable.
- Part 2 pixel tuning uses two-fold CV to control runtime, increasing selection variance.
- The stroke dataset is small and highly imbalanced; metrics are educational and not clinical evidence.
- H2O searches are expensive and have not been rerun after the evaluation fixes.
- SuperCon elemental-composition grouping assumes the two UCI CSV files retain their documented row alignment; the loader verifies row count and target equality because no explicit row ID exists.
- Elemental composition does not capture crystal structure, phase, pressure, defects, or synthesis conditions.
- Target-conditioned SuperCon bucket metrics cannot be compared with end-to-end predictor performance.

## Project Evolution

This repository began as a multi-part course submission. It was later reorganized
to remove import-time execution, make paths and seeds configurable, isolate
generated outputs, correct label semantics and evaluation leakage, and add an
editable documentation layer suitable for an undergraduate portfolio. Original
PDFs and selected figures remain available as clearly labeled historical context.
