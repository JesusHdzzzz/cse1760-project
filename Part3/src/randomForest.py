import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report
)
from sklearn.impute import SimpleImputer
import numpy as np

# 1. Load the Dataset
# Make sure the file is in the same directory as this script.
try:
    df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
except FileNotFoundError:
    print("Error: 'healthcare-dataset-stroke-data.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# Drop the 'id' column as it is not a feature
df = df.drop('id', axis=1)

# 2. Data Cleaning and Preprocessing

# A. Handle 'gender' category 'Other' (which has only 1 observation)
df = df[df['gender'] != 'Other']

# B. Impute missing 'bmi' values with the mean
imputer = SimpleImputer(strategy='mean')
df['bmi'] = imputer.fit_transform(df[['bmi']])

# C. One-Hot Encode Categorical Features
df_encoded = pd.get_dummies(
    df,
    columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'],
    drop_first=True
)

# D. Define Features (X) and Target (y)
X = df_encoded.drop('stroke', axis=1)
y = df_encoded['stroke']

# 3. Split Data into Training and Testing Sets
# Use a stratified split because the target variable ('stroke') is imbalanced
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")
print("-" * 50)

# 4. Train the Random Forest Classifier
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced', # Crucial for imbalanced datasets
    max_depth=10,
    min_samples_split=5
)

print("Training Random Forest Classifier...")
rf_classifier.fit(X_train, y_train)
print("Training complete.")
print("-" * 50)

# 5. Model Prediction and Probability
y_pred = rf_classifier.predict(X_test)
# Get probabilities for the positive class (stroke=1) for AUC/AUCPR calculation
y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]

# 6. Calculate Metrics

# A. AUC and AUCPR
auc_roc = roc_auc_score(y_test, y_pred_proba)
auc_pr = average_precision_score(y_test, y_pred_proba)

# B. Classification Report Metrics (as a dictionary)
# The labels are integer 0 and 1, so the keys in the dictionary will be string '0' and '1'.
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

# Extract metrics for the positive class (class '1' - Stroke)
precision_stroke = report['1']['precision']
recall_stroke = report['1']['recall']
f1_stroke = report['1']['f1-score']

# Extract Macro Average F1-score
macro_f1 = report['macro avg']['f1-score']

print("Model Evaluation Metrics:")
print("-" * 50)

# Print AUC and AUCPR
print(f"Area Under ROC Curve (AUC): {auc_roc:.4f}")
print(f"Area Under Precision-Recall Curve (AUCPR): {auc_pr:.4f}")

print("\nMetrics for Positive Class (Stroke = 1):")
print(f"Precision (Class 1): {precision_stroke * 100:.2f}%")
print(f"Recall (Sensitivity) (Class 1): {recall_stroke * 100:.2f}%")
print(f"F1-score (Class 1): {f1_stroke * 100:.2f}%")

print("\nOverall Averages:")
print(f"Macro Average F1-score: {macro_f1 * 100:.2f}%")
print("-" * 50)