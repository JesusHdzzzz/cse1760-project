import pandas as pd
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
import warnings
import time
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore', category=FutureWarning)

start_time = time.perf_counter()

df = pd.read_csv("data/healthcare-dataset-stroke-data.csv")

bmi_median = df['bmi'].median()
df['bmi'].fillna(bmi_median, inplace=True)

df = df[df['gender'] != 'Other']

h2o.init(max_mem_size="8G")

h2o_df = h2o.H2OFrame(df)

y = 'stroke'
X = h2o_df.columns
X.remove('id')
X.remove(y)

h2o_df[y] = h2o_df[y].asfactor()

train, test = h2o_df.split_frame(ratios=[0.8], seed = 1234)

print("\n--- Class Distribution in Splits ---")
print("\nTraining Set:")
train_dist = train[y].table()
train_counts = train_dist.as_data_frame()
print(f"No Stroke: {int(train_counts.iloc[0, 1])}, Stroke: {int(train_counts.iloc[1, 1])}")

print("\nTest Set:")
test_dist = test[y].table()
test_counts = test_dist.as_data_frame()
print(f"No Stroke: {int(test_counts.iloc[0, 1])}, Stroke: {int(test_counts.iloc[1, 1])}")

totalnostroke = int(train_counts.iloc[0, 1]) + int(test_counts.iloc[0, 1])
totalstroke = int(train_counts.iloc[1, 1]) + int(test_counts.iloc[1, 1])
print(f"Totals: No Stroke: {totalnostroke}, Stroke: {totalstroke}")

hyper_params = {
    'max_depth': [4,],
    'learn_rate': [0.05],
    'ntrees': [70],
    'sample_rate': [0.7],
    'col_sample_rate': [0.9],
    'min_rows': [2]
}

gbm_base = H2OGradientBoostingEstimator(
    seed=1234,
    balance_classes=True,
    class_sampling_factors=[1.0, 11.0],
    nfolds=2,
    keep_cross_validation_predictions=True,
    fold_assignment="Stratified"
)

grid = H2OGridSearch(
    model=gbm_base,
    hyper_params=hyper_params,
    search_criteria={'strategy': "Cartesian"}
)

grid.train(x=X, y=y, training_frame=train)

rows = []
for m in grid.models:
    ntrees = m.actual_params["ntrees"]
    test_aucpr = m.model_performance(test_data=test).aucpr()
    rows.append((ntrees, test_aucpr))

best_by_ntrees = {}
for ntrees, aucpr in rows:
    best_by_ntrees[ntrees] = max(best_by_ntrees.get(ntrees, -1), aucpr)

nt_list = sorted(best_by_ntrees.keys())
aucpr_list = [best_by_ntrees[n] for n in nt_list]

plt.figure(figsize=(10, 6))
plt.plot(nt_list, aucpr_list, marker='o', linewidth=2)
plt.xlabel("Number of Trees (ntrees)")
plt.ylabel("Test AUCPR")
plt.title("Test AUCPR vs Number of Trees")
plt.grid(True, alpha=0.3)
plt.tight_layout()
#plt.savefig('aucpr_vs_ntrees.png')
plt.show()

grid_perf = grid.get_grid(sort_by="aucpr", decreasing=True)

print("\n--- Top Models from Grid Search (Sorted by AUCPR) ---")
print(grid_perf)

best_gbm = grid_perf.models[0]

cv_pred = best_gbm.cross_validation_holdout_predictions()
cv_pred_df = cv_pred.as_data_frame()
cv_true = train[y].as_data_frame()[y].values
cv_scores = cv_pred_df["p1"].values

thresholds = np.linspace(0.01, 0.99, 200)

best_f2 = -1
best_threshold = None
best_cm = None

for t in thresholds:
    y_pred = (cv_scores >= t).astype(int)

    tn, fp, fn, tp = confusion_matrix(cv_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f2 = 0
    if precision + recall > 0:
        f2 = (5 * precision * recall) / (4 * precision + recall)
    else:
        f2 = 0

    if f2 > best_f2:
        best_f2 = f2
        best_threshold = t
        best_cm = (tn, fp, fn, tp)

test_pred = best_gbm.predict(test).as_data_frame()["p1"].values
y_test = test[y].as_data_frame()[y].values
y_test_pred = (test_pred >= best_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred, labels=[0, 1]).ravel()

precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / (tp + tn + fp + fn)

print("\n--- Best F2 Threshold ---")
print(f"Threshold: {best_threshold:.3f}")
print(f"F2 Score: {best_f2:.4f}")

print("\n--- Confusion Matrix at F2 Threshold ---")
print(f"TN: {tn}, FP: {fp}")
print(f"FN: {fn}, TP: {tp}")

print("\n--- Metrics ---")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Accuracy: {accuracy:.4f}")

prec, rec, pr_thresholds = precision_recall_curve(y_test, test_pred)
aucpr = average_precision_score(y_test, test_pred)

plt.figure(figsize=(8, 6))
plt.plot(rec, prec, linewidth=2, label=f"AUCPR = {aucpr:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (Best GBM)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
#plt.savefig('simple_aucpr.png')
plt.show()

fpr, tpr, roc_thresholds = roc_curve(y_test, test_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)  # chance line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Best GBM)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
#plt.savefig('roc_curve.png')
plt.show()

print(f"Best model test AUCPR: {aucpr:.4f}")
print(f"Best model test AUC:   {roc_auc:.4f}")

print(f"\n--- Best Model Hyperparameters ---")
print(f"Max Depth: {best_gbm.actual_params['max_depth']}, Learning Rate: {best_gbm.actual_params['learn_rate']}, Number of Trees: {best_gbm.actual_params['ntrees']}, Sample Rate: {best_gbm.actual_params['sample_rate']}, Col Sample Rate: {best_gbm.actual_params['col_sample_rate']}, Min Rows: {best_gbm.actual_params['min_rows']}")

end_time = time.perf_counter()
print("\ntime taken: ", (end_time - start_time))

h2o.cluster().shutdown()