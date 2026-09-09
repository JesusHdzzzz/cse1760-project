# Part 2: MNIST XGBoost Experiments

Part 2 compares XGBoost classifiers trained on raw MNIST pixels and precomputed
LeNet features. Both experiments use a pipeline containing `StandardScaler`,
PCA retaining 80% variance, and `XGBClassifier`, so fold-specific preprocessing
is learned only from each training fold.

```bash
python part2/src/xgb_mnist_pixels.py --help
python part2/src/xgb_mnist_lenet.py --help

python part2/src/xgb_mnist_pixels.py --output-dir part2/outputs/pixels-corrected-labels
python part2/src/xgb_mnist_lenet.py --output-dir part2/outputs/lenet-corrected-labels
```

Put `MNIST.mat` and `MNIST-LeNet5.mat` in `part2/data/`; see the root README for
expected shapes and checksums. The loader implements the course MAT convention:
raw `10` is digit 0, and raw `1` through `9` keep their semantic identities.

The pixel search uses two stratified folds as a documented coursework runtime
tradeoff; the LeNet search uses three. Hyperparameters are selected by CV on the
training split, tree count is selected on validation, the full 60,000-row training
set is used for the final fit, and the 10,000-row test set is evaluated once.

The origin of the 800 LeNet features is incompletely documented. The repository
does not include the feature-extraction code, architecture details beyond the
filename, training data, or checkpoint. Treat them as externally supplied fixed
features, not as an independently reproducible representation-learning pipeline.

All files under `part2/results/` and `part2/reports/` predate the corrected label
mapping. They are historical/stale and must not be quoted as results from the
current implementation. New runs write to ignored `part2/outputs/` directories.
