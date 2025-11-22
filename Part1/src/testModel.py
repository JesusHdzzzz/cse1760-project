import scipy.io
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load and filter data
mat_data = scipy.io.loadmat('../images/MNISTmini.mat')
images = mat_data['train_fea1']
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

print(f"\n=== FILTERED DATASET ===")
print(f"Images of 5s and 6s: {len(X)}")

# Use specific ranges for train/validation/test
train_indices = range(0, 1000)      # Images 0-999 (first 1000)
val_indices = range(1000, 2000)     # Images 1000-1999 (next 1000)  
test_indices = range(2000, 3000)    # Images 2000-2999 (next 1000)

# Split the data
X_train = X[train_indices]
y_train = y[train_indices]

X_val = X[val_indices]
y_val = y[val_indices]

X_test = X[test_indices]
y_test = y[test_indices]

print(f"\n=== MANUAL SPLIT ===")
print(f"Training set: {len(X_train)} images (indices 0-999)")
print(f"Validation set: {len(X_val)} images (indices 1000-1999)")
print(f"Test set: {len(X_test)} images (indices 2000-2999)")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Find best C using validation set (instead of cross-validation)
best_score = 0
best_C = 1
best_model = None

for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    model = LogisticRegression(penalty='l2', C=C, solver='liblinear', random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on validation set
    val_score = model.score(X_val_scaled, y_val)
    
    print(f"C={C}: Validation accuracy = {val_score:.4f}")
    
    if val_score > best_score:
        best_score = val_score
        best_C = C
        best_model = model

print(f"\n=== BEST MODEL SELECTION ===")
print(f"Best C: {best_C}, Best validation accuracy: {best_score:.4f}")

# Train final model on combined training + validation data for best performance
print(f"\n=== FINAL TRAINING ===")
print("Training final model on combined training + validation data...")

# Combine training and validation sets
X_combined = np.vstack([X_train, X_val])
y_combined = np.hstack([y_train, y_val])

# Scale the combined data
scaler_final = StandardScaler()
X_combined_scaled = scaler_final.fit_transform(X_combined)
X_test_scaled_final = scaler_final.transform(X_test)

# Train final model on combined data
final_model = LogisticRegression(penalty='l2', C=best_C, solver='liblinear', random_state=42)
final_model.fit(X_combined_scaled, y_combined)

# Evaluate on test set
test_accuracy = final_model.score(X_test_scaled_final, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Optional: Compare with model trained only on training data
val_model = LogisticRegression(penalty='l2', C=best_C, solver='liblinear', random_state=42)
val_model.fit(X_train_scaled, y_train)
val_test_accuracy = val_model.score(X_test_scaled, y_test)

print(f"Test accuracy (trained only on training set): {val_test_accuracy:.4f}")
print(f"Improvement from using combined data: {test_accuracy - val_test_accuracy:.4f}")