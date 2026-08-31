# src/xgb_mnist_pixels_fixed.py
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from utils_mnist import load_mnist_mat, train_val_split
from pathlib import Path

PART2_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PART2_DIR / "results" / "pixels" 
DATA_DIR = PART2_DIR / "data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    start_time = time.time()

    print("Loading MNIST pixel data...")
    X_all, y_all = load_mnist_mat(
        DATA_DIR / "MNIST.mat",
        feature_key="train_fea",
        label_key="train_gnd"
    )
    X_test, y_test = load_mnist_mat(
        DATA_DIR / "MNIST.mat",
        feature_key="test_fea",
        label_key="test_gnd"
    )

    print(f"Original shapes - Train: {X_all.shape}, Test: {X_test.shape}")
    print(f"Memory estimate for training features: {X_all.nbytes / (1024**3):.2f} GB")

    X_train, X_val, y_train, y_val = train_val_split(X_all, y_all)
    print(f"Split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

 
    print("\n=== Applying PCA for Dimensionality Reduction ===")


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)


    pca = PCA(n_components=0.80, random_state=42, svd_solver='full')
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"PCA reduced pixel features from {X_train.shape[1]} → {X_train_pca.shape[1]}")
    print(f"Variance retained: {np.sum(pca.explained_variance_ratio_):.4f}")

    print("\n=== PCA Feature Selection Explained ===")
    print(f"784 original pixel dimensions → {X_train_pca.shape[1]} PCA components.")

    X_full_pca = np.vstack([X_train_pca, X_val_pca])
    y_full = np.concatenate([y_train, y_val])

    print("\n=== Optimized XGBoost Parameter Search ===")

    base_model = XGBClassifier(
        learning_rate=0.1,
        objective="multi:softmax",
        num_class=10,
        eval_metric="mlogloss",
        n_jobs=-1,
        tree_method="hist",
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
    )

    param_dist = {
        "n_estimators": [100, 150],
        "max_depth": [4, 5, 6],
        "colsample_bytree": [0.5, 0.7],
        "subsample": [0.8, 0.9],
        "reg_lambda": [0.1, 1],
    }

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=4,
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    print("Starting optimized parameter search...")
    search_start = time.time()
    random_search.fit(X_train_pca, y_train)
    search_time = time.time() - search_start

    print(f"\nParameter search finished in {search_time:.1f} seconds")
    print(f"Best CV accuracy: {random_search.best_score_:.4f}")
    print(f"Best parameters: {random_search.best_params_}")

    best_params = random_search.best_params_


    print("\n=== Generating Test Error vs Trees Plot ===")

    tree_counts = [50, 100, 150, 200]
    test_errors = []

    print("Training models for plot…")
    for n_trees in tree_counts:
        model = XGBClassifier(
            n_estimators=n_trees,
            max_depth=best_params["max_depth"],
            learning_rate=0.1,
            colsample_bytree=best_params.get("colsample_bytree", 0.8),
            subsample=best_params.get("subsample", 0.8),
            reg_lambda=best_params.get("reg_lambda", 1),
            objective="multi:softmax",
            num_class=10,
            eval_metric="mlogloss",
            n_jobs=-1,
            tree_method="hist",
            random_state=42,
            use_label_encoder=False,
            verbosity=0,
        )

        model.fit(X_full_pca, y_full)
        y_pred = model.predict(X_test_pca)
        test_error = 1 - accuracy_score(y_test, y_pred)
        test_errors.append(test_error)

        print(f"  Trees={n_trees}, Test Error={test_error:.4f}")


    plt.figure(figsize=(10, 6))
    plt.plot(tree_counts, test_errors, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Number of Trees')
    plt.ylabel('Test Error')
    plt.title(f"XGBoost on MNIST Pixel PCA Features (max_depth={best_params['max_depth']})")
        
 
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.5)

    plt.savefig(OUTPUT_DIR / "pixels_test_error_vs_trees.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "pixels_test_error_vs_trees.eps", format="eps", bbox_inches="tight")
    plt.close()

    print("\n✓ Saved plots:")
    print("  - pixels_test_error_vs_trees.pdf")
    print("  - pixels_test_error_vs_trees.eps")


    print("\n=== Training Final Best Model ===")

    best_model = random_search.best_estimator_
    
    """
    print("Training best model with early stopping…")
    train_start = time.time()

    best_model.fit(
        X_full_pca, y_full,
        eval_set=[(X_val_pca, y_val)],
        verbose=False
    )

    train_time = time.time() - train_start
    print(f"Training completed in {train_time:.1f} seconds")
    """
    
    print("✓ Best model already trained and ready for evaluation")


    print("\n=== Saving Model and Results ===")

    with open(OUTPUT_DIR / "best_pixels_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open(OUTPUT_DIR / "pixels_pca_transformer.pkl", "wb") as f:
        pickle.dump({"pca": pca, "scaler": scaler}, f)

    y_pred = best_model.predict(X_test_pca)
    test_acc = accuracy_score(y_test, y_pred)
    best_test_error = 1 - test_acc

    params_text = "\n".join([f"- {k}: {v}" for k, v in best_params.items()])

    results_text = f"""test error {best_test_error*100:.2f}%, {best_params['n_estimators']} trees, maximum depth {best_params['max_depth']}
Other hyperparameters:
{params_text}
PCA components: {X_train_pca.shape[1]} (80% variance retained)
"""

    with open(OUTPUT_DIR / "pixels_results.txt", "w") as f:
        f.write(results_text)

    print(" Saved: best_pixels_model.pkl")
    print(" Saved: pixels_pca_transformer.pkl")
    print(" Saved: pixels_results.txt")


    print("\n=== Final Evaluation ===")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Error:    {best_test_error:.4f}")
    print(f"Runtime:       {time.time() - start_time:.1f} seconds")

    # Feature importance
    print("\n=== Feature Importance Analysis ===")
    importances = best_model.feature_importances_
    top_n = min(10, len(importances))
    top_idx = np.argsort(importances)[-top_n:][::-1]

    print(f"Top {top_n} PCA components:")
    for idx in top_idx:
        print(f"  Component {idx}: importance={importances[idx]:.4f}, "
              f"variance={pca.explained_variance_ratio_[idx]:.4f}")

    # Hardest digits
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    hardest = np.argsort(np.diag(cm) / cm.sum(axis=1))[:3]
    print(f"Hardest digits to classify: {hardest}")


    print("\n" + "=" * 60)
    print("RESULTS FOR SUBMISSION:")
    print("=" * 60)
    print(results_text.strip())
    print("=" * 60)

    print("\n" + "=" * 60)
    print("FILES GENERATED:")
    print("=" * 60)
    print("1. pixels_test_error_vs_trees.pdf")
    print("2. pixels_test_error_vs_trees.eps")
    print("3. best_pixels_model.pkl")
    print("4. pixels_pca_transformer.pkl")
    print("5. pixels_results.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()