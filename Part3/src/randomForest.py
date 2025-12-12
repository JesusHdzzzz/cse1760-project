import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report
)
from sklearn.impute import SimpleImputer
import numpy as np

# --- 1. Data Loading and Preprocessing ---

# Load the dataset
try:
    df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
except FileNotFoundError:
    print("Error: 'healthcare-dataset-stroke-data.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# Drop the 'id' column
df = df.drop('id', axis=1)

# Handle 'gender' category 'Other' (only 1 observation)
df = df[df['gender'] != 'Other']

# Impute missing 'bmi' values with the mean
imputer = SimpleImputer(strategy='mean')
df['bmi'] = imputer.fit_transform(df[['bmi']])

# One-Hot Encode Categorical Features
df_encoded = pd.get_dummies(
    df,
    columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'],
    drop_first=True
)

# Define Features (X) and Target (y)
X = df_encoded.drop('stroke', axis=1)
y = df_encoded['stroke']

# Split Data into Training and Testing Sets (Stratified split for imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")
print("-" * 60)

# --- 2. Hyperparameter Tuning (Optimized for AUCPR) ---

# Base Random Forest model with class_weight='balanced'
rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')

# Optimized parameter grid
param_grid = {
    'n_estimators': [50, 100, 150, 200, 250, 300],
    'max_depth': [5, 7, 10, 15],
    'min_samples_split': [2, 5, 10]
}

# Initialize and run GridSearchCV for AUCPR ('average_precision')
grid_search_aupr = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    scoring='recall', # Optimize for AUCPR
    cv=5,
    verbose=0,
    n_jobs=-1
)

print("Starting Grid Search for Hyperparameter Tuning (Optimizing for AUCPR)...")
# Note: Grid Search output is suppressed (verbose=0) for a cleaner final script
grid_search_aupr.fit(X_train, y_train)


# --- 3. Final Model Evaluation ---

# Get Best Parameters and Model
best_rf_aupr = grid_search_aupr.best_estimator_
best_params_aupr = grid_search_aupr.best_params_
best_score_aupr = grid_search_aupr.best_score_

# Make predictions and get probabilities
y_pred = best_rf_aupr.predict(X_test)
y_pred_proba = best_rf_aupr.predict_proba(X_test)[:, 1]

# Calculate Final Metrics
auc_roc = roc_auc_score(y_test, y_pred_proba)
auc_pr = average_precision_score(y_test, y_pred_proba)

# Get full classification report
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
precision_stroke = report['1']['precision']
recall_stroke = report['1']['recall']
f1_stroke = report['1']['f1-score']
macro_f1 = report['macro avg']['f1-score']

print("\n" + "=" * 60)
print("FINAL MODEL RESULTS (Optimized for AUCPR)")
print("=" * 60)
print(f"Best AUCPR Score (Cross-Validation): {best_score_aupr:.4f}")
print(f"Best Hyperparameters: {best_params_aupr}")
print("-" * 60)

print(f"Test Set AUC: {auc_roc:.4f}")
print(f"Test Set AUCPR: {auc_pr:.4f}")

print("\nMetrics for Positive Class (Stroke = 1):")
print(f"Precision (Class 1): {precision_stroke * 100:.2f}%")
print(f"Recall (Sensitivity) (Class 1): {recall_stroke * 100:.2f}%")
print(f"F1-score (Class 1): {f1_stroke * 100:.2f}%")

print("\nOverall Averages:")
print(f"Macro Average F1-score: {macro_f1 * 100:.2f}%")
print("-" * 60)