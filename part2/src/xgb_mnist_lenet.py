import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from utils_mnist import load_mnist_mat, train_val_split

RANDOM_STATE = 42 
PCA_VARIANCE = 0.80 
TREE_COUNTS = [50, 100, 150, 200]

PART2_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PART2_DIR / "data"
OUTPUT_DIR = PART2_DIR / "results" / "lenet" 

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    start_time = time.time()
    
    print("="*60)
    print("XGBoost Classification for MNIST-LeNet5 Features")
    print("="*60)
    

    print("\n[1/7] Loading MNIST-LeNet5 data...")
    X_train_all, y_train_all = load_mnist_mat(
        DATA_DIR / "MNIST-LeNet5.mat",
        feature_key="train_fea",
        label_key="train_gnd"
    )
    X_test, y_test = load_mnist_mat(
        DATA_DIR / "MNIST-LeNet5.mat",
        feature_key="test_fea",
        label_key="test_gnd"
    )
    
    print(f"   Training data: {X_train_all.shape} samples, {X_train_all.shape[1]} features")
    print(f"   Test data: {X_test.shape} samples")
    print(f"   Unique labels in training: {np.unique(y_train_all)}")
    print(f"   Unique labels in test: {np.unique(y_test)}")
    print(f"   Labels converted from 1-10 to 0-9")
    

    print("\n[2/7] Splitting data into train/validation sets...")
    X_train, X_val, y_train, y_val = train_val_split(
        X_train_all,
        y_train_all,
        random_state=RANDOM_STATE,
    )
    
    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Validation set: {X_val.shape[0]} samples")
    
    print("\n[3/7] Building preprocessing/model pipeline")
   
    pipeline = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "pca",
            PCA(
                n_components=PCA_VARIANCE,
                svd_solver="full",
            ),
        ),
        (
            "model",
            XGBClassifier(
                learning_rate=0.1,
                objective="multi:softmax",
                num_class=10,
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=0,
            ),
        ),
    ])
    
    
    print("\n[4/7] Hyperparameter tuning with RandomizedSearchCV...")

    
    param_dist = {
        "model__max_depth": [4, 6, 8],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__subsample": [0.6, 0.8, 1.0],
    }
    
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=6,
        cv=StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=RANDOM_STATE,
        ),
        scoring="accuracy",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
        error_score="raise",
    )
    
    print("   Starting randomized search...")
    search_start = time.time()
    
    random_search.fit(X_train, y_train)
    
    search_time = time.time() - search_start
    print(f"   Search completed in {search_time:.1f} seconds")
    print(f"   Best CV accuracy: {random_search.best_score_:.4f}")
    print(f"   Best parameters: {random_search.best_params_}")
    
    best_params = random_search.best_params_

    print("\n[5/7] Selecting optimal number of trees using validation set...")

    val_errors = []

    for n_trees in TREE_COUNTS:
        candidate = clone(random_search.best_estimator_)

        candidate.set_params(
            model__n_estimators=n_trees,
        )

        candidate.fit(X_train, y_train)

        y_val_pred = candidate.predict(X_val)

        val_error = 1 - accuracy_score(
            y_val,
            y_val_pred,
        )

        val_errors.append(val_error)

        print(
            f"   Trees: {n_trees:3d}, "
            f"Validation Error: {val_error:.4f}"
        )

    best_idx = np.argmin(val_errors)
    best_tree_count = TREE_COUNTS[best_idx]
    best_val_error = val_errors[best_idx]

    print(
        f"\n   Selected tree count: {best_tree_count} "
        f"(validation error: {best_val_error:.4f})"
    )

    # Create validation-error plot.
    plt.figure(figsize=(10, 6))

    plt.plot(
        TREE_COUNTS,
        val_errors,
        marker="o",
        linewidth=2,
        label="Validation Error",
    )

    plt.axvline(
        x=best_tree_count,
        linestyle=":",
        linewidth=2,
        label=f"Selected ({best_tree_count} trees)",
    )

    plt.xlabel("Number of Trees")
    plt.ylabel("Validation Error")
    plt.title("XGBoost on MNIST LeNet5 Features")

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(
        OUTPUT_DIR / "lenet5_validation_error_vs_trees.pdf",
        format="pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        OUTPUT_DIR / "lenet5_validation_error_vs_trees.eps",
        format="eps",
        bbox_inches="tight",
    )

    plt.close()

    print("   Saved validation-error plots.")

    print("\n[6/7] Training final model on full training data...")

    final_model = clone(
        random_search.best_estimator_
    )

    final_model.set_params(
        model__n_estimators=best_tree_count,
        model__n_jobs=-1,
    )

    final_model.fit(
        X_train_all,
        y_train_all,
    )

    print(
        f"   Final model trained with "
        f"{best_tree_count} trees."
    )

    # Test set is used only after all model choices are finished.
    y_pred_final = final_model.predict(X_test)

    test_acc = accuracy_score(
        y_test,
        y_pred_final,
    )

    final_test_error = 1 - test_acc

    print(
        f"   Final test accuracy: "
        f"{test_acc:.4f}"
    )

    print(
        f"   Final test error: "
        f"{final_test_error:.4f}"
    )

    print("\n[7/7] Saving model and results...")

    with open(
        OUTPUT_DIR / "best_lenet5_model.pkl",
        "wb",
    ) as f:
        pickle.dump(final_model, f)

    print("   Saved: best_lenet5_model.pkl")

    final_pca = final_model.named_steps["pca"]

    pca_components = final_pca.n_components_

    clean_params = {
        key.replace("model__", ""): value
        for key, value in best_params.items()
    }

    params_text = "\n".join(
        f"- {key}: {value}"
        for key, value in clean_params.items()
    )

    results_text = (
        f"Final test accuracy: {test_acc:.4f}\n"
        f"Final test error: {final_test_error * 100:.2f}%\n"
        f"Selected trees: {best_tree_count}\n"
        f"Best CV accuracy: {random_search.best_score_:.4f}\n"
        f"Best validation error: {best_val_error:.4f}\n"
        f"\n"
        f"Best hyperparameters:\n"
        f"{params_text}\n"
        f"\n"
        f"PCA components: {pca_components}\n"
        f"PCA variance retained: {PCA_VARIANCE:.0%}\n"
    )

    with open(
        OUTPUT_DIR / "lenet5_results.txt",
        "w",
    ) as f:
        f.write(results_text)

    print("   Saved: lenet5_results.txt")

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"Total runtime: "
        f"{total_time:.1f} seconds"
    )

    print(
        f"Final test accuracy: "
        f"{test_acc:.4f}"
    )

    print(
        f"Final test error: "
        f"{final_test_error:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred_final,
            digits=4,
        )
    )

    print("\nConfusion Matrix:")

    cm = confusion_matrix(
        y_test,
        y_pred_final,
    )

    print(
        "Rows: True labels, "
        "Columns: Predicted labels"
    )

    print(cm)

    print(
        "\nMost confused digit pairs "
        "(top 5):"
    )

    confusions = []

    for true_label in range(10):
        for predicted_label in range(10):

            if (
                true_label != predicted_label
                and cm[
                    true_label,
                    predicted_label,
                ] > 0
            ):
                confusions.append(
                    (
                        true_label,
                        predicted_label,
                        cm[
                            true_label,
                            predicted_label,
                        ],
                    )
                )

    confusions.sort(
        key=lambda item: item[2],
        reverse=True,
    )

    for (
        true_label,
        predicted_label,
        count,
    ) in confusions[:5]:

        print(
            f"   {true_label} -> "
            f"{predicted_label}: "
            f"{count} misclassifications"
        )

    print("\nFiles generated:")

    print(
        "1. "
        "lenet5_validation_error_vs_trees.pdf"
    )

    print(
        "2. "
        "lenet5_validation_error_vs_trees.eps"
    )

    print("3. best_lenet5_model.pkl")
    print("4. lenet5_results.txt")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    
    print(results_text)
if __name__ == "__main__":
    main()