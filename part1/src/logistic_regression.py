from pathlib import Path 

import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from data_utils import (
    load_mnist,
    filter_digits,
    encode_binary_labels,
    split_data,
)

X, y = load_mnist()
X, y = filter_digits(X, y, digits=(5,6))

X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    X,
    y,
    train_size=1000,
    val_size=1000,
    test_size=1000,
    random_state=42,
)

y_train = encode_binary_labels(y_train, positive_class=6) 
y_val = encode_binary_labels(y_val, positive_class=6)
y_test = encode_binary_labels(y_test, positive_class=6)



C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
cv_scores_mean = []
cv_scores_std = []
train_accs = []
val_accs = []
best_C = None
best_cv = 0.0

print("\n" + "=" * 60)
print("Testing different C values with 5-fold cross-validation")
print("=" * 60)

for C_val in C_values:

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=C_val,
            solver="liblinear",
            max_iter=1000,
            random_state=42,
        )),
    ])
    
  
    cv_results = cross_val_score(
        model,
        X_train, 
        y_train,
        cv=5,
        scoring="accuracy",
    )
    
    cv_mean = np.mean(cv_results)
    cv_std = np.std(cv_results)
    
  
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train) 
    val_acc = model.score(X_val, y_val)
    
    
    cv_scores_mean.append(cv_mean)
    cv_scores_std.append(cv_std)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    print(f"C={C_val:7.3f} -> Train: {train_acc:.4f}, CV: {cv_mean:.4f} (±{cv_std:.4f}), Val: {val_acc:.4f}")
    
   
    if cv_mean > best_cv:
        best_cv = cv_mean
        best_C = C_val

print(f"\nBest C value: {best_C} (CV accuracy = {best_cv:.4f})")


train_errs = [1 - acc for acc in train_accs]
val_errs = [1 - acc for acc in val_accs]

GRAPH_DIR = Path(__file__).resolve().parent.parent / "graphs" 
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 6))
plt.semilogx(C_values, train_errs, 'b-', marker='o', label='Training Error', linewidth=2)
plt.semilogx(C_values, val_errs, 'r-', marker='s', label='Validation Error', linewidth=2)
plt.axvline(x=best_C, color='g', linestyle='--', label=f'Best C = {best_C}')
plt.xlabel('Regularization Parameter (C) - log scale')
plt.ylabel('Error Rate')
plt.title('Logistic Regression: Training vs Validation Error\n(Best C selected via 5-fold CV)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().invert_yaxis()
plt.savefig(GRAPH_DIR / 'lr_training_vs_validation_error.png', dpi=300, bbox_inches='tight')
plt.show()


print(f"\n" + "=" * 60)
print("Training final model")
print("=" * 60)
print(f"Using C={best_C} on combined training+validation data")


X_combined = np.vstack([X_train, X_val])
y_combined = np.concatenate([y_train, y_val])


scaler_final = StandardScaler()
X_combined_scaled = scaler_final.fit_transform(X_combined)


final_model = LogisticRegression(
    C=best_C,
    penalty='l2',
    solver='liblinear',
    max_iter=1000,
    random_state=42
)
final_model.fit(X_combined_scaled, y_combined)


X_test_scaled_final = scaler_final.transform(X_test)
predictions = final_model.predict(X_test_scaled_final)
test_acc = accuracy_score(y_test, predictions)


combined_train_acc = final_model.score(X_combined_scaled, y_combined)

print(f"\n" + "=" * 60)
print("Results")
print("=" * 60)
best_idx = C_values.index(best_C)
print(f"Best C value: {best_C}")
print(f"Training Accuracy: {train_accs[best_idx]:.4f}")
print(f"Cross-Validation Accuracy: {best_cv:.4f}")
print(f"Validation Accuracy: {val_accs[best_idx]:.4f}")
print(f"Final Model Training Accuracy (train+val): {combined_train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")


train_err = 1 - train_accs[best_idx]
val_err = 1 - val_accs[best_idx]
test_err = 1 - test_acc

print(f"\n" + "=" * 60)
print("Error Analysis")
print("=" * 60)
print(f"Training Error: {train_err:.4f}")
print(f"Validation Error: {val_err:.4f}")
print(f"Test Error: {test_err:.4f}")
print(f"\nOverfitting gap (val - train): {val_err - train_err:.4f}")
print(f"Generalization gap (test - val): {test_err - val_err:.4f}")

print(f"\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"{'Metric':<30} {'Value':<10}")
print("-" * 40)
print(f"{'Best C':<30} {best_C:<10.4f}")
print(f"{'Training Accuracy':<30} {train_accs[best_idx]:<10.4f}")
print(f"{'CV Accuracy':<30} {best_cv:<10.4f}")
print(f"{'Validation Accuracy':<30} {val_accs[best_idx]:<10.4f}")
print(f"{'Test Accuracy':<30} {test_acc:<10.4f}")
print(f"{'Final Training Accuracy':<30} {combined_train_acc:<10.4f}")