# Part 1: MNIST Digits 5 vs 6

Part 1 compares logistic regression and random forest classification on the
100-feature `MNISTmini.mat` representation. Defaults use deterministic,
stratified 1,000-row train, validation, and test splits with seed 42.

```bash
python part1/src/feature.py --help
python part1/src/logistic_regression.py --help
python part1/src/random_forest.py --help

python part1/src/feature.py
python part1/src/logistic_regression.py
python part1/src/random_forest.py
```

Place `MNISTmini.mat` in `part1/data/`. The reviewed local copy has SHA-256
`c33d5d8782df9f080d9efd1bb9823a58c87d53686c2970f5e70aa3c9c3db05ba`
and contains `train_fea1` (60,000 x 100), `train_gnd1` (60,000 x 1),
`test_fea1` (10,000 x 100), and `test_gnd1` (10,000 x 1).

Model selection uses five-fold stratified CV on training data. Validation scores
are supplementary diagnostics and do not select hyperparameters. The chosen model
is refit on train plus validation and tested once. Candidate training/validation
scores therefore describe a different fitted model from the final test score;
the scripts do not report their difference as a generalization gap.

[Current curated rerun results](../results/part1/) are published separately from
the generated output tree.

New figures and JSON metadata are written to `part1/outputs/`. Files under
`graphs/` and `reports/` are historical course artifacts and are not current
results for the refactored scripts.
