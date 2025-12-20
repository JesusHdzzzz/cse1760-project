import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, classification_report, accuracy_score)
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import time

try:
    df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
except FileNotFoundError:
    print("Error: 'healthcare-dataset-stroke-data.csv' not found. Please ensure the file is in the correct directory.")
    exit()

start_time = time.perf_counter()

df = df.drop('id', axis=1)

df = df[df['gender'] != 'Other']

imputer = SimpleImputer(strategy='median')
df['bmi'] = imputer.fit_transform(df[['bmi']])

df_encoded = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)

X = df_encoded.drop('stroke', axis=1)
y = df_encoded['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')

param_grid = {
    'n_estimators': [50],
    'max_depth': [5],
    'min_samples_split': [5]
}

grid_search_aupr = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    scoring='average_precision',
    cv=2,
    verbose=0,
    n_jobs=-1
)

grid_search_aupr.fit(X_train, y_train)

best_rf_aupr = grid_search_aupr.best_estimator_
best_params_aupr = grid_search_aupr.best_params_
best_score_aupr = grid_search_aupr.best_score_

y_pred = best_rf_aupr.predict(X_test)
y_pred_proba = best_rf_aupr.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)

auc_roc = roc_auc_score(y_test, y_pred_proba)
auc_pr = average_precision_score(y_test, y_pred_proba)

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
precision_stroke = report['1']['precision']
recall_stroke = report['1']['recall']


print(f"Best AUCPR Score (Cross-Validation): {best_score_aupr:.4f}")
print(f"Best Hyperparameters: {best_params_aupr}")

print(f"Test Set AUC: {auc_roc:.4f}")
print(f"Test Set AUCPR: {auc_pr:.4f}")

print("\nMetrics for Positive Class (Stroke = 1):")
print(f"Precision: {precision_stroke * 100:.2f}%")
print(f"Recall: {recall_stroke * 100:.2f}%")
print(f"Accuracy: {accuracy * 100:.2f}%")

end_time = time.perf_counter()
print("\ntime taken: ", (end_time - start_time))


n_estimators_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]


test_aucpr_scores = []

for n in n_estimators_list:
    rf = RandomForestClassifier(
        n_estimators=n,
        max_depth=best_params_aupr['max_depth'],
        min_samples_split=best_params_aupr['min_samples_split'],
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    y_proba = rf.predict_proba(X_test)[:, 1]
    
    aucpr = average_precision_score(y_test, y_proba)
    test_aucpr_scores.append(aucpr)

plt.figure()
plt.plot(n_estimators_list, test_aucpr_scores, marker='o')

plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Test AUCPR')
plt.title('Test AUCPR vs Number of Trees')
plt.grid(True)
plt.savefig('aucpr_vs_nestimators.png')
plt.show()
