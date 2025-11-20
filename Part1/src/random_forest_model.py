# src/random_forest_model.py

import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load data
mat_data = scipy.io.loadmat('images/MNISTmini.mat')

# The .mat file should contain at least these:
#   train_fea1: features (images as 1D vectors)
#   train_gnd1: labels (digits 0–9)
images = mat_data['train_fea1']
labels = mat_data['train_gnd1'].flatten()

# 2. Filter to digits 5 and 6 only (binary classification)
mask = (labels == 5) | (labels == 6)
X = images[mask]
y_digits = labels[mask]

# Map digits to binary labels (0 for 5, 1 for 6)
y = np.where(y_digits == 5, 0, 1)

# Basic dataset stats
total_5 = np.sum(y_digits == 5)
total_6 = np.sum(y_digits == 6)
total_images = len(y_digits)

print("=== DATASET OVERVIEW (5 vs 6) ===")
print(f"Total images (5 or 6): {total_images}")
print(f"Images of digit 5: {total_5}")
print(f"Images of digit 6: {total_6}")
print()

# 3. Train/test split (same style as logistic regression)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])
print()

# 4. Cross-validate over number of trees (n_estimators)
n_estimators_list = [10, 50, 100, 200, 500]
best_n = None
best_cv_score = 0.0

print("=== Cross-validation over n_estimators ===")
for n in n_estimators_list:
    rf = RandomForestClassifier(
        n_estimators=n,
        criterion='gini',       # default
        max_depth=None,         # default (nodes expanded until all leaves are pure)
        max_features='sqrt',    # default for classifiers
        n_jobs=-1,              # use all cores
        random_state=42
    )
    scores = cross_val_score(rf, X_train, y_train, cv=5)
    mean_score = scores.mean()
    std_score = scores.std()
    
    print(f"n_estimators = {n:3d}  ->  CV mean accuracy = {mean_score:.4f} (+/- {std_score:.4f})")
    
    if mean_score > best_cv_score:
        best_cv_score = mean_score
        best_n = n

print()
print(f"Best n_estimators: {best_n} with CV accuracy {best_cv_score:.4f}")
print()

# 5. Train final RandomForest on full training set with best_n
final_rf = RandomForestClassifier(
    n_estimators=best_n,
    criterion='gini',
    max_depth=None,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)
final_rf.fit(X_train, y_train)

# 6. Evaluate on train and test sets
y_train_pred = final_rf.predict(X_train)
y_test_pred = final_rf.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("=== FINAL MODEL PERFORMANCE ===")
print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy:  {test_acc:.4f}")
print()

print("=== Test set classification report (0 = digit 5, 1 = digit 6) ===")
print(classification_report(y_test, y_test_pred))

print("=== Test set confusion matrix ===")
print(confusion_matrix(y_test, y_test_pred))