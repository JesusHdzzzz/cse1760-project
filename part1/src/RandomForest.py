import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

from pathlib import Path

from data_utils import (
    load_mnist, 
    filter_digits, 
    encode_binary_labels,
    split_data
)




DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "MNISTmini.mat"
mat = scipy.io.loadmat(DATA_PATH)

X, y = load_mnist()
X, y = filter_digits(X, y, digits=(5,6))


print(f"Got {len(X)} images of 5 and 6")


X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, 1000, 1000, 1000, 42)


n_options = [10, 50, 100, 200, 500]
cv_scores_avg = []
cv_scores_std = []
train_accs = []
val_accs = []

best_n_val = None
best_score = 0

print("\nTrying different tree counts...")

for n_trees in n_options:
    clf = RandomForestClassifier(
        n_estimators=n_trees,
        random_state=42,
        n_jobs=-1 
    )
    

    scores = cross_val_score(clf, X_train, y_train, cv=5)
    cv_avg = scores.mean()
    cv_std = scores.std()
    

    clf.fit(X_train, y_train)
    

    train_acc = clf.score(X_train, y_train)
    val_acc = clf.score(X_val, y_val)
    
    print(f"n={n_trees}: CV={cv_avg:.3f}, Train={train_acc:.3f}, Val={val_acc:.3f}")
    
    cv_scores_avg.append(cv_avg)
    cv_scores_std.append(cv_std)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
   
    if cv_avg > best_score:
        best_score = cv_avg
        best_n_val = n_trees

print(f"\nBest is n={best_n_val} with CV score {best_score:.4f}")


train_err = [1 - a for a in train_accs]
val_err = [1 - a for a in val_accs]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(n_options, train_err, 'b-o', linewidth=2, label='Training Error')
ax.plot(n_options, val_err, 'r-s', linewidth=2, label='Validation Error')
ax.axvline(x=best_n_val, color='green', linestyle='--', alpha=0.7, label=f'Best n={best_n_val}')

ax.set_xlabel('Number of Trees')
ax.set_ylabel('Error Rate')
ax.set_title('Training vs Validation Error (Random Forest)')
ax.grid(True, alpha=0.3)
ax.legend()
ax.invert_yaxis() 
plt.tight_layout()
plt.savefig('rf_error_plot.png', dpi=200)
plt.show()


print("\nMaking final model with train+val...")
X_combined = np.vstack([X_train, X_val])
y_combined = np.concatenate([y_train, y_val])

final_model = RandomForestClassifier(
    n_estimators=best_n_val,
    random_state=42,
    n_jobs=4
)
final_model.fit(X_combined, y_combined)

# Test it
preds = final_model.predict(X_test)
test_acc = accuracy_score(y_test, preds)

# Results
print("\n" + "="*50)
print("RESULTS")
print("="*50)

best_idx = n_options.index(best_n_val)
print(f"Best n_estimators: {best_n_val}")
print(f"Training accuracy: {train_accs[best_idx]:.4f}")
print(f"CV accuracy:       {best_score:.4f}")
print(f"Validation acc:    {val_accs[best_idx]:.4f}")
print(f"Test accuracy:     {test_acc:.4f}")

# Some error analysis
train_error_rate = 1 - train_accs[best_idx]
val_error_rate = 1 - val_accs[best_idx]
test_error_rate = 1 - test_acc

print(f"\nTraining error:   {train_error_rate:.4f}")
print(f"Validation error: {val_error_rate:.4f}")
print(f"Test error:       {test_error_rate:.4f}")

# Check for overfitting
if train_accs[best_idx] - val_accs[best_idx] > 0.05:
    print("\nNote: Possible overfitting (train much higher than validation)")
elif test_acc < val_accs[best_idx] - 0.05:
    print("\nNote: Test performance worse than validation")
else:
    print("\nLooks good! Model generalizes well.")

