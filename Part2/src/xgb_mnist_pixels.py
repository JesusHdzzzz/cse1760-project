# src/xgb_mnist_pixels.py
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from utils_mnist import load_mnist_mat, train_val_split

def main():
    X_all, y_all = load_mnist_mat(
        "images/MNIST.mat", 
        feature_key="train_fea", 
        label_key="train_gnd"
    )
    X_test, y_test = load_mnist_mat(
        "images/MNIST.mat", 
        feature_key="test_fea", 
        label_key="test_gnd"
    )

    print(f"Train features: {X_all.shape}")
    print(f"Test features: {X_test.shape}")

    X_train, X_val, y_train, y_val = train_val_split(X_all, y_all)

    print(f"Train size: {X_train.shape[0]}")
    print(f"Val   size: {X_val.shape[0]}")

    n_list = [50, 100, 200]
    depth_list = [4, 6, 8]

    best_score = 0
    best_model = None
    best_params = None

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    print("\n=== XGBoost Cross-Validation (Pixel Features) ===")
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

            print(f"n={n}, depth={depth} → CV Accuracy = {mean_acc:.4f}")

            if mean_acc > best_score:
                best_score = mean_acc
                best_model = model
                best_params = (n, depth)

    print(f"\nBest: n_estimators={best_params[0]}, max_depth={best_params[1]} "
          f"(CV={best_score:.4f})")

    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    best_model.fit(X_full, y_full)

    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\nFinal Test Accuracy (pixel features): {test_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()