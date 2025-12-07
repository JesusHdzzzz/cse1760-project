#!/usr/bin/env python3
import scipy.io
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# =========================
# 1. Load and filter data
# =========================
mat_data = scipy.io.loadmat('images/MNISTmini.mat')  # adjust path if needed
images = mat_data['train_fea1']          # (N, 100)
labels = mat_data['train_gnd1'].flatten()

mask = (labels == 5) | (labels == 6)
X = images[mask]
y = np.where(labels[mask] == 5, 0, 1)  # Binary labels

total_5 = np.sum(labels == 5)
total_6 = np.sum(labels == 6)
total_images = len(labels)

print(f"=== DATASET OVERVIEW ===")
print(f"Total images in dataset: {total_images}")
print(f"Images of digit 5: {total_5} ({total_5/total_images*100:.1f}%)")
print(f"Images of digit 6: {total_6} ({total_6/total_images*100:.1f}%)")
print(f"Total 5s + 6s: {total_5 + total_6} ({(total_5 + total_6)/total_images*100:.1f}% of dataset)")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# Helper to compute train/val accuracy for a given model
def eval_model(model, Xtr, ytr, Xval, yval):
    model.fit(Xtr, ytr)
    train_acc = model.score(Xtr, ytr)
    val_acc   = model.score(Xval, yval)
    return train_acc, val_acc


# ==========================================================
# 3. Sweep 1: C (inverse regularization strength, L2 penalty)
# ==========================================================
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
train_err_C = []
val_err_C   = []

print("\n=== Sweep: C (L2, solver='liblinear') ===")
for C in C_values:
    logreg = LogisticRegression(
        penalty='l2',
        C=C,
        solver='liblinear',
        random_state=42,
        max_iter=1000
    )
    tr_acc, val_acc = eval_model(logreg, X_train_s, y_train, X_val_s, y_val)
    train_err_C.append(1 - tr_acc)
    val_err_C.append(1 - val_acc)
    print(f"C={C:7} | train_acc={tr_acc:.4f}, val_acc={val_acc:.4f}")

plt.figure(figsize=(6, 4))
plt.semilogx(C_values, train_err_C, marker='o', label='Train error')
plt.semilogx(C_values, val_err_C, marker='o', label='Val error')
plt.xlabel('C (inverse regularization strength)')
plt.ylabel('Error = 1 - accuracy')
plt.title('Logistic Regression: Error vs C (L2, liblinear)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('output/logreg_error_vs_C.png', dpi=300)


# =====================================================
# 4. Sweep 2: Penalty type (L1 vs L2) with fixed C
# =====================================================
penalties = ['l1', 'l2']
C_fixed = 1.0
train_err_pen = []
val_err_pen   = []

print("\n=== Sweep: penalty (C=1.0, solver='liblinear') ===")
for pen in penalties:
    logreg = LogisticRegression(
        penalty=pen,
        C=C_fixed,
        solver='liblinear',  # supports both L1 and L2
        random_state=42,
        max_iter=1000
    )
    tr_acc, val_acc = eval_model(logreg, X_train_s, y_train, X_val_s, y_val)
    train_err_pen.append(1 - tr_acc)
    val_err_pen.append(1 - val_acc)
    print(f"penalty={pen:2} | train_acc={tr_acc:.4f}, val_acc={val_acc:.4f}")

x = np.arange(len(penalties))
plt.figure(figsize=(5, 4))
plt.bar(x - 0.15, train_err_pen, width=0.3, label='Train error')
plt.bar(x + 0.15, val_err_pen,   width=0.3, label='Val error')
plt.xticks(x, penalties)
plt.ylabel('Error = 1 - accuracy')
plt.title('Logistic Regression: Error vs Penalty (C=1.0)')
plt.legend()
plt.tight_layout()
plt.savefig('output/logreg_error_vs_penalty.png', dpi=300)


# ===========================================================
# 5. Sweep 3: Solver choice (liblinear vs lbfgs, L2, C fixed)
# ===========================================================
solvers = ['liblinear', 'lbfgs']
C_fixed = 1.0
train_err_solver = []
val_err_solver   = []

print("\n=== Sweep: solver (L2, C=1.0) ===")
for solver in solvers:
    logreg = LogisticRegression(
        penalty='l2',
        C=C_fixed,
        solver=solver,
        random_state=42,
        max_iter=1000
    )
    tr_acc, val_acc = eval_model(logreg, X_train_s, y_train, X_val_s, y_val)
    train_err_solver.append(1 - tr_acc)
    val_err_solver.append(1 - val_acc)
    print(f"solver={solver:9} | train_acc={tr_acc:.4f}, val_acc={val_acc:.4f}")

x = np.arange(len(solvers))
plt.figure(figsize=(5, 4))
plt.bar(x - 0.15, train_err_solver, width=0.3, label='Train error')
plt.bar(x + 0.15, val_err_solver,   width=0.3, label='Val error')
plt.xticks(x, solvers)
plt.ylabel('Error')
plt.title('Logistic Regression: Error vs Solver (L2, C=1.0)')
plt.legend()
plt.tight_layout()
plt.savefig('output/logreg_error_vs_solver.png', dpi=300)


# ======================================================
# 6. Final model selection and TEST performance
#    (choose best C from Sweep 1 with L2/liblinear)
# ======================================================
best_idx = int(np.argmin(val_err_C))
best_C = C_values[best_idx]
print(f"\nBest C from Sweep 1: {best_C} with val_err={val_err_C[best_idx]:.4f}")

# Train final model on Train+Val with best C
X_trainval_s = np.vstack([X_train_s, X_val_s])
y_trainval = np.concatenate([y_train, y_val])

final_logreg = LogisticRegression(
    penalty='l2',
    C=best_C,
    solver='liblinear',
    random_state=42,
    max_iter=1000
)
final_logreg.fit(X_trainval_s, y_trainval)
test_acc = final_logreg.score(X_test_s, y_test)
print(f"Final TEST accuracy (L2, liblinear, C={best_C}): {test_acc:.4f}")

#plt.show()  # show all figures when running interactively