import scipy.io
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# === CHOOSE YOUR DIGIT PAIR HERE ===
DIGIT_A = 5
DIGIT_B = 6
# ===================================

# Load .mat file
mat = scipy.io.loadmat("../images/MNISTmini.mat")

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
needed_per_digit = 4000

# Take the first 1500 of each (you could also shuffle)
idx_a = idx_a[:needed_per_digit]
idx_b = idx_b[:needed_per_digit]

def split_indices(idx):
    """Split 1500 indices into 500 train, 500 val, 500 test."""
    train_idx = idx[:1000]
    val_idx   = idx[1000:2000]
    test_idx  = idx[2000:3000]
    return train_idx, val_idx, test_idx

a_train, a_val, a_test = split_indices(idx_a)
b_train, b_val, b_test = split_indices(idx_b)

# Build splits
train_idx = np.concatenate([a_train, b_train])
val_idx   = np.concatenate([a_val, b_val])
test_idx  = np.concatenate([a_test, b_test])

# Optionally shuffle within each split
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

# Convert y labels to binary: 0 for DIGIT_A, 1 for DIGIT_B
def make_binary_labels(y):
    return np.where(y == DIGIT_A, 0, 1)

y_train_bin = make_binary_labels(y_train)
y_val_bin   = make_binary_labels(y_val)
y_test_bin  = make_binary_labels(y_test)

print(f"\nUsing digits: {DIGIT_A} vs {DIGIT_B}")
print("Train size:", X_train.shape[0])
print("Val   size:", X_val.shape[0])
print("Test  size:", X_test.shape[0])

# -------------------------------
# Random Forest + cross-validation
# -------------------------------

n_list = [10, 50, 100, 200, 500]
best_n = None
best_score = 0.0

print("\nCross-validation over n_estimators:")

for n in n_list:
    rf = RandomForestClassifier(
        n_estimators=n,
        max_depth=None,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )
    scores = cross_val_score(rf, X_train, y_train_bin, cv=5)
    mean = scores.mean()
    print(f"{n:4d} trees -> CV accuracy = {mean:.4f}")

    if mean > best_score:
        best_score = mean
        best_n = n

print(f"\nBest n_estimators: {best_n} (CV accuracy = {best_score:.4f})")

# Train final model on Train + Val
X_final = np.vstack([X_train, X_val])
y_final = np.concatenate([y_train_bin, y_val_bin])

rf_final = RandomForestClassifier(
    n_estimators=best_n,
    max_depth=None,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)
rf_final.fit(X_final, y_final)

# Evaluate on Test
y_pred = rf_final.predict(X_test)

acc = accuracy_score(y_test_bin, y_pred)
print("\nFinal Test Accuracy:", acc)
print("\nClassification Report (0 = digit", DIGIT_A, ", 1 = digit", DIGIT_B, "):")
print(classification_report(y_test_bin, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test_bin, y_pred))