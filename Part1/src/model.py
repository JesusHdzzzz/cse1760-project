import scipy.io
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Find best C using cross-validation
best_score = 0
best_C = 1
for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    model = LogisticRegression(penalty='l2', C=C, solver='liblinear', random_state=42)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    mean_score = scores.mean()
    if mean_score > best_score:
        best_score = mean_score
        best_C = C

print(f"Best C: {best_C}, Best CV score: {best_score:.4f}")

# Train final model
final_model = LogisticRegression(penalty='l2', C=best_C, solver='liblinear', random_state=42)
final_model.fit(X_train_scaled, y_train)

# Evaluate
test_accuracy = final_model.score(X_test_scaled, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")