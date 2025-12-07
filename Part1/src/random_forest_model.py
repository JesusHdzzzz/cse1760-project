#!/usr/bin/env python3
import scipy.io
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

DIGIT_A = 5
DIGIT_B = 6

# Load .mat file
mat = scipy.io.loadmat("images/MNISTmini.mat")

X_all = mat['train_fea1']              # (60000, 100)
y_all = mat['train_gnd1'].flatten()    # (60000,)

print("X_all shape:", X_all.shape)
print("y_all shape:", y_all.shape)

# Get indices for each chosen digit
idx_a = np.where(y_all == DIGIT_A)[0]
idx_b = np.where(y_all == DIGIT_B)[0]

print(f"Found {len(idx_a)} samples of digit {DIGIT_A}")
print(f"Found {len(idx_b)} samples of digit {DIGIT_B}")

# We want at least 1500 per digit (500 train, 500 val, 500 test)
needed_per_digit = 1500
if len(idx_a) < needed_per_digit or len(idx_b) < needed_per_digit:
    raise ValueError(
        f"Not enough samples for one of the digits. "
        f"Need {needed_per_digit} each, got {len(idx_a)} (digit {DIGIT_A}) "
        f"and {len(idx_b)} (digit {DIGIT_B})."
    )

# Take the first 1500 of each (you could also shuffle)
idx_a = idx_a[:needed_per_digit]
idx_b = idx_b[:needed_per_digit]

def split_indices(idx):
    """Split 1500 indices into 500 train, 500 val, 500 test."""
    train_idx = idx[:500]
    val_idx   = idx[500:1000]
    test_idx  = idx[1000:1500]
    return train_idx, val_idx, test_idx

a_train, a_val, a_test = split_indices(idx_a)
b_train, b_val, b_test = split_indices(idx_b)

train_idx = np.concatenate([a_train, b_train])
val_idx   = np.concatenate([a_val, b_val])
test_idx  = np.concatenate([a_test, b_test])

rng = np.random.default_rng(42)
rng.shuffle(train_idx)
rng.shuffle(val_idx)
rng.shuffle(test_idx)

X_train = X_all[train_idx]
y_train = y_all[train_idx]
X_val   = X_all[val_idx]
y_val   = y_all[val_idx]
X_test  = X_all[test_idx]
y_test  = y_all[test_idx]

def make_binary(y):
    return np.where(y == DIGIT_A, 0, 1)

y_train_b = make_binary(y_train)
y_val_b   = make_binary(y_val)
y_test_b  = make_binary(y_test)

print("Train size:", X_train.shape[0])
print("Val size  :", X_val.shape[0])
print("Test size :", X_test.shape[0])

# Helper to evaluate RF and return train/val error
def eval_rf(**rf_kwargs):
    rf = RandomForestClassifier(
        n_estimators=rf_kwargs.get('n_estimators', 100),
        max_depth=rf_kwargs.get('max_depth', None),
        min_samples_leaf=rf_kwargs.get('min_samples_leaf', 1),
        max_features=rf_kwargs.get('max_features', 'sqrt'),
        bootstrap=rf_kwargs.get('bootstrap', True),
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train_b)
    tr_acc = rf.score(X_train, y_train_b)
    val_acc = rf.score(X_val, y_val_b)
    return rf, 1 - tr_acc, 1 - val_acc


# ======================================
# 2. Sweep 1: n_estimators
# ======================================
n_list = [10, 50, 100, 200, 500]
train_err_n = []
val_err_n   = []

print("\n=== Sweep: n_estimators ===")
for n in n_list:
    _, tr_err, val_err = eval_rf(n_estimators=n)
    train_err_n.append(tr_err)
    val_err_n.append(val_err)
    print(f"n={n:4d} | train_err={tr_err:.4f}, val_err={val_err:.4f}")

plt.figure(figsize=(6, 4))
plt.plot(n_list, train_err_n, marker='o', label='Train error')
plt.plot(n_list, val_err_n, marker='o', label='Val error')
plt.xlabel('n_estimators')
plt.ylabel('Error = 1 - accuracy')
plt.title('Random Forest: Error vs n_estimators')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('output/rf_error_vs_n_estimators.png', dpi=300)

# pick best n for later
best_n = n_list[int(np.argmin(val_err_n))]


# ======================================
# 3. Sweep 2: max_depth (using best_n)
# ======================================
depth_list = [None, 5, 10, 20]
train_err_d = []
val_err_d   = []

print("\n=== Sweep: max_depth (n_estimators = best_n) ===")
for d in depth_list:
    _, tr_err, val_err = eval_rf(n_estimators=best_n, max_depth=d)
    train_err_d.append(tr_err)
    val_err_d.append(val_err)
    print(f"max_depth={str(d):>4} | train_err={tr_err:.4f}, val_err={val_err:.4f}")

x_axis = [d if d is not None else 0 for d in depth_list]  # map None->0 for plotting
plt.figure(figsize=(6, 4))
plt.plot(x_axis, train_err_d, marker='o', label='Train error')
plt.plot(x_axis, val_err_d, marker='o', label='Val error')
plt.xticks(x_axis, [str(d) for d in depth_list])
plt.xlabel('max_depth (None means unlimited)')
plt.ylabel('Error')
plt.title(f'Random Forest: Error vs max_depth (n={best_n})')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('output/rf_error_vs_max_depth.png', dpi=300)

best_depth = depth_list[int(np.argmin(val_err_d))]


# ======================================
# 4. Sweep 3: min_samples_leaf
# ======================================
leaf_list = [1, 2, 4, 8]
train_err_leaf = []
val_err_leaf   = []

print("\n=== Sweep: min_samples_leaf (n, depth fixed) ===")
for leaf in leaf_list:
    _, tr_err, val_err = eval_rf(
        n_estimators=best_n,
        max_depth=best_depth,
        min_samples_leaf=leaf
    )
    train_err_leaf.append(tr_err)
    val_err_leaf.append(val_err)
    print(f"min_samples_leaf={leaf:2d} | train_err={tr_err:.4f}, val_err={val_err:.4f}")

plt.figure(figsize=(6, 4))
plt.plot(leaf_list, train_err_leaf, marker='o', label='Train error')
plt.plot(leaf_list, val_err_leaf, marker='o', label='Val error')
plt.xlabel('min_samples_leaf')
plt.ylabel('Error')
plt.title(f'Random Forest: Error vs min_samples_leaf (n={best_n}, depth={best_depth})')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('output/rf_error_vs_min_samples_leaf.png', dpi=300)

best_leaf = leaf_list[int(np.argmin(val_err_leaf))]


# ======================================
# 5. Sweep 4: max_features
# ======================================
feat_list = ['sqrt', 'log2', None]
train_err_feat = []
val_err_feat   = []

print("\n=== Sweep: max_features (n, depth, leaf fixed) ===")
for feat in feat_list:
    _, tr_err, val_err = eval_rf(
        n_estimators=best_n,
        max_depth=best_depth,
        min_samples_leaf=best_leaf,
        max_features=feat
    )
    train_err_feat.append(tr_err)
    val_err_feat.append(val_err)
    print(f"max_features={str(feat):>4} | train_err={tr_err:.4f}, val_err={val_err:.4f}")

x = np.arange(len(feat_list))
plt.figure(figsize=(5, 4))
plt.bar(x - 0.15, train_err_feat, width=0.3, label='Train error')
plt.bar(x + 0.15, val_err_feat,   width=0.3, label='Val error')
plt.xticks(x, [str(f) for f in feat_list])
plt.ylabel('Error')
plt.title('Random Forest: Error vs max_features')
plt.legend()
plt.tight_layout()
plt.savefig('output/rf_error_vs_max_features.png', dpi=300)


# ======================================
# 6. Final model and TEST performance
# ======================================
print("\nBest hyperparameters based on validation error:")
print(f"  n_estimators    = {best_n}")
print(f"  max_depth       = {best_depth}")
print(f"  min_samples_leaf= {best_leaf}")
# choose best max_features:
best_feat = feat_list[int(np.argmin(val_err_feat))]
print(f"  max_features    = {best_feat}")

rf_final = RandomForestClassifier(
    n_estimators=best_n,
    max_depth=best_depth,
    min_samples_leaf=best_leaf,
    max_features=best_feat,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)
rf_final.fit(X_train, y_train_b)

y_pred_test = rf_final.predict(X_test)
test_acc = accuracy_score(y_test_b, y_pred_test)
print(f"\nFinal TEST accuracy: {test_acc:.4f}")
print("\nClassification report:")
print(classification_report(y_test_b, y_pred_test))
print("Confusion matrix:")
print(confusion_matrix(y_test_b, y_pred_test))

#plt.show()