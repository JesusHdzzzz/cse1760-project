import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load and filter data
mat_data = scipy.io.loadmat('../images/MNISTmini.mat')
images = mat_data['train_fea1']
labels = mat_data['train_gnd1'].flatten()

mask = (labels == 5) | (labels == 6)
X = images[mask]
y = np.where(labels[mask] == 5, 0, 1)  # Binary labels

# Split into train+validation and test
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Further split train+validation into train and validation
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2 of total

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter values to test
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
train_errors = []
val_errors = []

print("=== HYPERPARAMETER TUNING ===")
print(f"{'C':<10} {'Train Error':<15} {'Val Error':<15}")
print("-" * 40)

best_val_error = float('inf')
best_C = None
best_model = None

for C in C_values:
    # Train model
    model = LogisticRegression(penalty='l2', C=C, solver='liblinear', random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Calculate errors
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    
    train_error = 1 - accuracy_score(y_train, y_train_pred)
    val_error = 1 - accuracy_score(y_val, y_val_pred)
    
    train_errors.append(train_error)
    val_errors.append(val_error)
    
    print(f"{C:<10} {train_error:<15.4f} {val_error:<15.4f}")
    
    # Track best model based on validation error
    if val_error < best_val_error:
        best_val_error = val_error
        best_C = C
        best_model = model

# Plot the learning curves
plt.figure(figsize=(10, 6))
plt.semilogx(C_values, train_errors, 'b-', marker='o', label='Training Error', linewidth=2)
plt.semilogx(C_values, val_errors, 'r-', marker='s', label='Validation Error', linewidth=2)
plt.axvline(x=best_C, color='g', linestyle='--', label=f'Best C = {best_C}')
plt.xlabel('Regularization Strength (C) - log scale')
plt.ylabel('Error Rate')
plt.title('Training and Validation Errors vs Regularization Strength')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Evaluate best model on test set
y_test_pred = best_model.predict(X_test_scaled)
test_error = 1 - accuracy_score(y_test, y_test_pred)

print(f"\n=== BEST MODEL SELECTION ===")
print(f"Selected C: {best_C}")
print(f"Training error: {train_errors[C_values.index(best_C)]:.4f}")
print(f"Validation error: {best_val_error:.4f}")
print(f"Test error: {test_error:.4f}")

# Compare errors
print(f"\n=== ERROR COMPARISON ===")
print(f"Test error vs Training error: {test_error - train_errors[C_values.index(best_C)]:.4f} (higher by)")
print(f"Test error vs Validation error: {test_error - best_val_error:.4f} (higher by)")

# Additional analysis: Check for overfitting/underfitting
train_val_diff = train_errors[C_values.index(best_C)] - best_val_error
print(f"\n=== MODEL ASSESSMENT ===")
if test_error > best_val_error + 0.05:  # More than 5% difference
    print(" Warning: Potential overfitting (test error significantly higher than validation)")
elif test_error < best_val_error - 0.05:  # More than 5% difference
    print("Warning: Potential underfitting or lucky test set")
else:
    print("Good generalization: Test error close to validation error")
    
if train_errors[C_values.index(best_C)] < 0.1 and test_error > 0.2:
    print(" Strong indication of overfitting!")
elif train_errors[C_values.index(best_C)] > 0.3:
    print(" Model may be underfitting (high training error)")

# Optional: Also show accuracy scores for clarity
print(f"\n=== ACCURACY SCORES ===")
print(f"Training accuracy: {1 - train_errors[C_values.index(best_C)]:.4f}")
print(f"Validation accuracy: {1 - best_val_error:.4f}")
print(f"Test accuracy: {1 - test_error:.4f}")