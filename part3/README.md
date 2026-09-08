# Part 3: Tree-Based Modeling Experiments

Part 3 contains two stroke-classification experiments and three SuperCon
regression/analysis scripts. Commands below work from the repository root; input
defaults are anchored to `part3/data/`, and generated artifacts are written below
`part3/outputs/`.

## Setup

```bash
./setup.sh
source .venv/bin/activate
```

H2O requires a working Java runtime on `PATH`. The pinned versions in
`requirements.txt` record the reviewed environment; Python and Java themselves
must still be recorded when publishing final results.

## Scripts

Run `--help` before an experiment to inspect its options.

```bash
python part3/src/histogram_boxplot.py
python part3/src/randomForest.py
python part3/src/h2oStroke2.py
python part3/src/GBM_optimization.py
python part3/src/supercon_split_by_tc_gbm.py
```

`h2oStroke2.py` performs a Cartesian search over 2,430 configurations with
two-fold cross-validation. `GBM_optimization.py --run-gbm-sweep` and the bucketed
GBM script are also full training workloads; they are not smoke tests.

## Methodology Status

The scripts preserve the submitted experiments, but the following issues must be
resolved before their test metrics are presented as unbiased final estimates:

- `randomForest.py` fits BMI imputation and one-hot encoding before the split. It
  also uses the test set for the number-of-trees comparison plot.
- `h2oStroke2.py` imputes BMI before splitting, uses a non-stratified H2O holdout,
  and evaluates every grid model on the test set for the tree-count plot.
- `GBM_optimization.py` derives its target-outlier limits before splitting. When
  the optional sweep runs, every candidate is evaluated on the test set and the
  output is sorted by test RMSE.
- `supercon_split_by_tc_gbm.py` creates buckets from the true target and trains a
  separate model in each bucket. This is target-conditioned error analysis unless
  a deployable, feature-only routing rule is defined; it is not an end-to-end
  predictor for samples whose critical temperature is unknown.
- Two-fold cross-validation in the imbalanced stroke experiments can have high
  variance. Any change to fold count, repeated validation, or split policy will
  change the methodology and should be agreed before rerunning reported results.

No metrics in the repository were regenerated during the reproducibility audit.

## Data Inventory

The data files are tracked in Git and total about 15 MiB, which makes the current
checkout self-contained. Keeping them tracked is reasonable for this course
project only after the original source URL, retrieval date, and redistribution
license are recorded. Those provenance details are not present in the repository,
so replacing the files with a downloader would currently be guesswork.

Reviewed SHA-256 fingerprints:

```text
644d473b05d2797006bd94865e4f8bb057f0c721617911613c82c8fcfc707420  healthcare-dataset-stroke-data.csv
4dfb6e3a1f6ffd969e5a5e42f093c4800d1e2a6c8b1e309f8fcd9f23d86952f3  train.csv
b68ae6b55ea8581eff8b1ffba073a899db7e2d2f7f3b781bb0802f643f51e5f7  unique_m.csv
```

`train.csv` supplies engineered SuperCon features. `unique_m.csv` has matching
row count and material-composition fields but is not consumed by the current
scripts. Its intended join or analysis role should be documented before use.
