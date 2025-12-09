# src/xgb_mnist_lenet.py
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from utils_mnist import load_mnist_mat, train_val_split

import time

def main():
    total_start = time.perf_counter()

    # ====== Load Data ======
    load_start = time.perf_counter()
    X_all, y_all = load_mnist_mat(
        "images/MNIST-LeNet5.mat",
        feature_key="train_fea",
        label_key="train_gnd"
    )
    X_test, y_test = load_mnist_mat(
        "images/MNIST-LeNet5.mat",
        feature_key="test_fea",
        label_key="test_gnd"
    )
    load_end = time.perf_counter()

    print(f"Train features: {X_all.shape}")
    print(f"Test features: {X_test.shape}")

    X_train, X_val, y_train, y_val = train_val_split(X_all, y_all)
    print(f"Train size: {X_train.shape[0]}")
    print(f"Val   size: {X_val.shape[0]}")

    # ====== Cross-Validation ======
    n_list = [50, 100, 200]
    depth_list = [4, 6, 8]

    best_score = 0
    best_model = None
    best_params = None

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    print("\n=== XGBoost Cross-Validation (LeNet5 Features) ===")
    cv_start = time.perf_counter()

    for n in n_list:
        for depth in depth_list:
            model = XGBClassifier(
                n_estimators=n,
                max_depth=depth,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softmax",
                num_class=10,
                eval_metric="mlogloss",
                n_jobs=-1,
                tree_method="hist",
                random_state=42,
            )

            scores = cross_val_score(model, X_train, y_train, cv=skf)
            mean_acc = scores.mean()

            print(f"n={n}, depth={depth} → CV Acc = {mean_acc:.4f}")

            if mean_acc > best_score:
                best_score = mean_acc
                best_model = model
                best_params = (n, depth)

    cv_end = time.perf_counter()

    print(f"\nBest: n_estimators={best_params[0]}, max_depth={best_params[1]} "
          f"(CV={best_score:.4f})")

    # ====== Final Training ======
    train_start = time.perf_counter()
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    best_model.fit(X_full, y_full)
    train_end = time.perf_counter()

    # ====== Final Evaluation ======
    eval_start = time.perf_counter()
    y_pred = best_model.predict(X_test)
    eval_end = time.perf_counter()

    test_acc = accuracy_score(y_test, y_pred)
    print(f"\nFinal Test Accuracy (LeNet5): {test_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    total_end = time.perf_counter()

    # ====== TIMING SUMMARY ======
    print("\n===== TIMING SUMMARY (LeNet5) =====")
    print(f"Data loading time:       {load_end - load_start:.2f} sec")
    print(f"Cross-validation time:   {cv_end - cv_start:.2f} sec")
    print(f"Final training time:     {train_end - train_start:.2f} sec")
    print(f"Evaluation time:         {eval_end - eval_start:.4f} sec")
    print(f"TOTAL runtime:           {total_end - total_start:.2f} sec")

if __name__ == "__main__":
    main()
