# xgb_mnist_lenet_fixed.py
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from pathlib import Path

PART2_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PART2_DIR / "results" / "lenet" 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_mnist_mat(path, feature_key="train_fea", label_key="train_gnd"):
    """Load MNIST data from .mat file and FIX labels to be 0-9."""
    import scipy.io
    data = scipy.io.loadmat(path)
    X = data[feature_key]
    y = data[label_key].ravel()
    
    y = y - 1  
    
    return X, y

def main():
    start_time = time.time()
    
    print("="*60)
    print("XGBoost Classification for MNIST-LeNet5 Features")
    print("="*60)
    

    print("\n[1/6] Loading MNIST-LeNet5 data...")
    X_train_all, y_train_all = load_mnist_mat(
        "../data/MNIST-LeNet5.mat",
        feature_key="train_fea",
        label_key="train_gnd"
    )
    X_test, y_test = load_mnist_mat(
        "../data/MNIST-LeNet5.mat",
        feature_key="test_fea",
        label_key="test_gnd"
    )
    
    print(f"   Training data: {X_train_all.shape} samples, {X_train_all.shape[1]} features")
    print(f"   Test data: {X_test.shape} samples")
    print(f"   Unique labels in training: {np.unique(y_train_all)}")
    print(f"   Unique labels in test: {np.unique(y_test)}")
    print(f"   Labels converted from 1-10 to 0-9")
    

    print("\n[2/6] Splitting data into train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all,
        test_size=5000/60000,  
        random_state=42,
        stratify=y_train_all
    )
    
    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Validation set: {X_val.shape[0]} samples")
    
    print("\n[3/6] Applying dimensionality reduction...")
    n_features = X_train.shape[1]
    pca = None
    scaler = None
    
    if n_features > 100:
        print(f"   Features ({n_features}) > 100, applying PCA...")
        
      
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
    
        pca = PCA(n_components=0.80, random_state=42, svd_solver='full')
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        X_train_used = X_train_pca
        X_val_used = X_val_pca
        X_test_used = X_test_pca
        
        print(f"   Reduced to {X_train_used.shape[1]} components")
        print(f"   Variance retained: {np.sum(pca.explained_variance_ratio_):.4f}")
    else:
        print(f"   Features ({n_features}) <= 100, skipping PCA")
        X_train_used = X_train
        X_val_used = X_val
        X_test_used = X_test
   
  
    print("\n[4/6] Hyperparameter tuning with RandomizedSearchCV...")
    
    base_model = XGBClassifier(
        learning_rate=0.1,
        objective="multi:softmax",
        num_class=10,  
        eval_metric="mlogloss",
        n_jobs=2,
        tree_method="hist",
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
    )
    
    param_dist = {
        'n_estimators': [50, 100, 150],
        'max_depth': [4, 6, 8],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'subsample': [0.6, 0.8, 1.0],
    }
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=6,  
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=2,
        random_state=42,
        verbose=1,
        error_score='raise'
    )
    
    print("   Starting randomized search...")
    search_start = time.time()
    
    random_search.fit(X_train_used, y_train)
    
    search_time = time.time() - search_start
    print(f"   Search completed in {search_time:.1f} seconds")
    print(f"   Best CV accuracy: {random_search.best_score_:.4f}")
    print(f"   Best parameters: {random_search.best_params_}")
    
    best_params = random_search.best_params_
    

    print("\n[5/6] Selecting optimal number of trees using validation set...")
    
    tree_counts = [50, 100, 150, 200]
    val_errors = []
    test_errors_for_plot = []
    
    for n_trees in tree_counts:
        model = XGBClassifier(
            n_estimators=n_trees,
            max_depth=best_params['max_depth'],
            learning_rate=0.1,
            colsample_bytree=best_params.get('colsample_bytree', 0.8),
            subsample=best_params.get('subsample', 0.8),
            objective="multi:softmax",
            num_class=10,
            eval_metric="mlogloss",
            n_jobs=2,
            tree_method="hist",
            random_state=42,
            verbosity=0,
            use_label_encoder=False,
        )
        
        model.fit(X_train_used, y_train)
        
        y_val_pred = model.predict(X_val_used)
        val_error = 1 - accuracy_score(y_val, y_val_pred)
        val_errors.append(val_error)
        
        y_test_pred = model.predict(X_test_used)
        test_error = 1 - accuracy_score(y_test, y_test_pred)
        test_errors_for_plot.append(test_error)
        
        print(f"   Trees: {n_trees:3d}, Val Error: {val_error:.4f}, Test Error: {test_error:.4f}")
    
    best_idx = np.argmin(val_errors)
    best_tree_count = tree_counts[best_idx]
    print(f"\n     Selected tree count: {best_tree_count} (based on validation error: {val_errors[best_idx]:.4f})")
    

    plt.figure(figsize=(10, 6))
    plt.plot(tree_counts, test_errors_for_plot, 'bo-', linewidth=2, markersize=8, label='Test Error')
    plt.plot(tree_counts, val_errors, 'rs--', linewidth=2, markersize=8, label='Validation Error')
    plt.axvline(x=best_tree_count, color='g', linestyle=':', linewidth=2, label=f'Selected ({best_tree_count} trees)')
    plt.xlabel('Number of Trees')
    plt.ylabel('Error Rate')
    plt.title(f'XGBoost on MNIST-LeNet5 Features (max_depth={best_params["max_depth"]})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(OUTPUT_DIR / 'lenet5_test_error_vs_trees.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'lenet5_test_error_vs_trees.eps', format='eps', bbox_inches='tight')
    plt.close()
    
    print("   Saved: lenet5_test_error_vs_trees.pdf")
    print("   Saved: lenet5_test_error_vs_trees.eps")
    

    print("\n[6/6] Training final model on full training data...")
    

    X_full = np.vstack([X_train_used, X_val_used])
    y_full = np.concatenate([y_train, y_val])
    
    final_model = XGBClassifier(
        n_estimators=best_tree_count,
        max_depth=best_params['max_depth'],
        learning_rate=0.1,
        colsample_bytree=best_params.get('colsample_bytree', 0.8),
        subsample=best_params.get('subsample', 0.8),
        objective="multi:softmax",
        num_class=10,
        eval_metric="mlogloss",
        n_jobs=-1,
        tree_method="hist",
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
    )
    
    final_model.fit(X_full, y_full)
    
    print(f"     Final model trained with {best_tree_count} trees")

    y_pred_final = final_model.predict(X_test_used)
    test_acc = accuracy_score(y_test, y_pred_final)
    final_test_error = 1 - test_acc
    
    print(f"     Final test accuracy: {test_acc:.4f}")
    print(f"     Final test error: {final_test_error:.4f}")
    
    print("\n[7/7] Saving model and results...")
    

    with open(OUTPUT_DIR / 'best_lenet5_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    print("     Saved: best_lenet5_model.pkl")
    

    if pca is not None and scaler is not None:
        with open(OUTPUT_DIR / 'lenet5_transformer.pkl', 'wb') as f:
            pickle.dump({'pca': pca, 'scaler': scaler}, f)
        print("     Saved: lenet5_transformer.pkl")
    
    params_text = ""
    for key, value in best_params.items():
        params_text += f"- {key}: {value}\n"
    
    results_text = f"""test error {final_test_error*100:.2f}%, {final_model.n_estimators} trees, maximum depth {best_params['max_depth']}
Other hyperparameters:
{params_text.strip()}
"""
    
    if pca is not None:
        results_text += f"PCA components: {X_train_used.shape[1]} (80% variance retained)\n"
    
    with open('lenet5_results.txt', 'w') as f:
        f.write(results_text)
    print("  Saved: lenet5_results.txt")
    
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print(" TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Total runtime: {total_time:.1f} seconds")
    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Final test error: {final_test_error:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_final, digits=4))
    
    print("\nFILES GENERATED:")
    print("-"*40)
    print("1. lenet5_test_error_vs_trees.pdf")
    print("2. lenet5_test_error_vs_trees.eps")
    print("3. best_lenet5_model.pkl")

        # Confusion Matrix
    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred_final)
    
    # Print formatted matrix
    print("Rows: True labels, Columns: Predicted labels")
    print("     0    1    2    3    4    5    6    7    8    9")  # Header
    
    for i in range(10):
        row = f"{i}: "
        for j in range(10):
            row += f"{cm[i,j]:4d} "
        print(row)
    
    # Most confused pairs
    print("\nMost confused digit pairs (top 5):")
    confusions = []
    for i in range(10):
        for j in range(10):
            if i != j and cm[i, j] > 0:
                confusions.append((i, j, cm[i, j]))
    
    confusions.sort(key=lambda x: x[2], reverse=True)
    for i, j, count in confusions[:5]:
        print(f"  {i} → {j}: {count} misclassifications")

    if pca is not None:
        print("4. lenet5_transformer.pkl")
    print("5. lenet5_results.txt")
    
    print("\n" + "="*60)
    print("RESULTS FOR SUBMISSION:")
    print("="*60)
    print(results_text)
    print("="*60)

if __name__ == "__main__":
    main()