# Part 3: Applied Tree-Based Modeling

Part 3 contains stroke classification and SuperCon critical-temperature
regression. All commands use repository-relative defaults and write generated
artifacts below ignored `part3/outputs/` directories.

## Commands

```bash
python part3/src/stroke_random_forest.py --help
python part3/src/stroke_h2o_gbm.py --help
python part3/src/histogram_boxplot.py --help
python part3/src/supercon_gbm.py --help
python part3/src/supercon_split_by_tc_gbm.py --help

python part3/src/stroke_random_forest.py
python part3/src/stroke_h2o_gbm.py
python part3/src/histogram_boxplot.py
python part3/src/supercon_gbm.py --run-gbm-sweep
python part3/src/supercon_split_by_tc_gbm.py
```

H2O scripts require a compatible Java runtime. By default a script attaches to
an existing H2O cluster without shutting it down; if it starts a cluster, it shuts
that cluster down on exit. `--keep-h2o-cluster` leaves a script-started cluster
running intentionally. The current local startup smoke test passed with OpenJDK
25.0.1 and H2O 3.46.0.12; this is not a general Java compatibility claim.

## Stroke Methodology

The dataset has 5,110 rows, including 249 positive stroke labels and 201 missing
BMI values. Both scripts remove the single `gender=Other` row and perform a
stratified train/test split before preprocessing.

The random forest places median/mode imputation and one-hot encoding inside a
scikit-learn pipeline, including inside five-fold CV. The H2O experiment leaves
BMI values missing and uses H2O GBM's native missing-value split routing, so each
cross-validation model learns NA routing only from its fold-training rows. Its
grid uses stratified five-fold CV and chooses an F2 threshold from out-of-fold
training predictions. The random forest selects by average precision; the H2O
grid selects by H2O AUCPR. Reported sklearn curve summaries use average
precision, and the two names are not used interchangeably.
The test set is used only for final evaluation and plots.

## SuperCon Methodology

The paired UCI files are verified to have equal row counts and identical row-wise
targets. The 86 elemental amount columns (`H` through `Rn`) in `unique_m.csv`
are put in atomic-number order, normalized by their row sum, and rounded to 12
decimal places to form an elemental-composition grouping key. Deterministic,
approximately 70/15/15 train/validation/test splits are disjoint by this key, so
reordered or proportionally scaled formulas cannot cross splits. The key does
not use `material`, `critical_temp`, or other metadata. Elemental composition is
not complete material identity: it does not capture structure, phase, pressure,
defects, or synthesis conditions.

The main GBM experiment derives optional IQR outlier bounds from the initial
training targets. Only rows used for fitting are filtered; validation and final
test evaluation cover their complete group splits. Candidate selection uses
validation RMSE. The selected configuration is refit on train plus validation,
then evaluated once on the unfiltered test set.

`supercon_split_by_tc_gbm.py` is target-conditioned diagnostic analysis. Bucket
boundaries are derived from training targets, but every row is routed using its
true `critical_temp`. Its bucket metrics are not deployable, end-to-end prediction
performance and should not be compared directly with the main GBM result.

## Data

- `healthcare-dataset-stroke-data.csv`: 5,110 x 12, from the [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset), SHA-256 `644d473b05d2797006bd94865e4f8bb057f0c721617911613c82c8fcfc707420`.
- `train.csv`: 21,263 x 82, UCI Superconductivity Data, SHA-256 `4dfb6e3a1f6ffd969e5a5e42f093c4800d1e2a6c8b1e309f8fcd9f23d86952f3`.
- `unique_m.csv`: 21,263 x 88, paired material compositions/formulas, SHA-256 `b68ae6b55ea8581eff8b1ffba073a899db7e2d2f7f3b781bb0802f643f51e5f7`.

The SuperCon source is [UCI dataset 464](https://archive.ics.uci.edu/dataset/464/superconductivty%2Bdata),
DOI `10.24432/C53P47`, licensed CC BY 4.0. The stroke Kaggle page says the data
is for educational use and should be credited to its author.

Current Part 3 experiments require reruns after the methodology changes. No
historical metric is presented as a current estimate.

The H2O stroke Cartesian grid contains 2,430 configurations before CV folds and
is the most expensive workflow in the repository. It is intentionally not a
smoke test.
