# Logistic Regression and Random Forest Classification on MNISTmini (Digits 5 vs 6)

## Logistic Regression Model Explanation

To classify our data, we decided on using the Logistic Regression model from scikit-learn. 
Before training the model, we needed to clean our data and preprocess it so the model could actually use it.

We first examined the `.mat` file that had the data on MATLAB Desktop, and saw that there are 
four sub “datasets,” each called `train_fea1`, `train_gnd1`, `test_fea1`, and `test_gnd1`.  
`train_fea1` contains the feature vectors that we use as input to our training model, and `train_gnd1` contains the 
corresponding class labels of each of the `train_fea1` vectors. The same goes for `test_fea1` and `test_gnd1`, except instead of being used for training, they are used for testing.

We then used this information and created a script to examine the shape of the data using 
NumPy’s built-in `.shape()` function, which we appropriately named `feature.py`. We can see that the 
`train_fea1` data is organized in a 2D array of shape `(60000, 100)`; this is a matrix where each row is an image. 
This is fine, and we can pass it into our model as is. However, our `train_gnd1` data is shown to have 
a 2D shape as well; more specifically, `(60000, 1)`. This is a natural result of how arrays 
are stored in MATLAB, but most Python machine learning libraries require ground-truth label 
arrays to be in 1D, so we use `.flatten()` to convert the label array into a proper vector.

The next step of our data processing is to filter out only the data we need. Because our 
problem is a binary classification task consisting only of digits **5 and 6**, we need 
to extract the appropriate images from the dataset. To do that, we defined a Boolean mask on 
our labels array. A mask is a Boolean array with a defined condition on a 
corresponding array; if the condition is true for a particular index, the 
corresponding position in the mask is `True`, and `False` otherwise. Our mask’s 
defined condition is: “if the label is 5 or 6, return True, otherwise False.”  
Applying this mask to our `train_fea1` feature matrix filters the dataset down to only images with labels 5 or 6.  
We also filtered the label array, and assigned digit **5 → negative class (0)**  
and digit **6 → positive class (1)**.  
Some print statements were added for basic dataset inspection.  
Finally, we standardized the data to have mean 0 and standard deviation 1.

Next comes the fun part: training the model. We want to prevent overfitting, so we include 
L2 regularization, which penalizes large weight values. This keeps the model’s decision boundary 
simpler and improves generalization. The most important hyperparameter in logistic regression is the `C` value, 
which is the inverse of the regularization strength. A larger `C` means weaker regularization (risking overfitting), 
while a smaller `C` means stronger regularization (risking underfitting). We want a value that balances both.

To find the optimal `C` value, we train our model multiple times using different candidate values, 
and evaluate each one using **5-fold cross-validation**. We keep track of the value of `C` that produces 
the highest average validation accuracy. Once we identify the best `C`, we retrain logistic regression 
on the full training data and evaluate it on the test set. The script then prints the final accuracy score.

**Sources:**
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html  
- https://www.ml-science.com/masking  
- https://stackoverflow.com/questions/47216388/efficiently-creating-masks-numpy-python  
- https://developers.google.com/machine-learning/crash-course/overfitting/regularization  


---

## Random Forest Model Explanation

For our second classifier, we implemented a **Random Forest** model using scikit-learn’s `RandomForestClassifier`. Although the preprocessing steps were similar to those used for logistic regression, Random Forests have different requirements and naturally handle raw, unscaled input features.

After loading the MNISTmini `.mat` file, we again flattened the label arrays to convert them from MATLAB’s `(N, 1)` format to a Python-friendly 1D vector. The feature matrix `train_fea1` contains **60,000** training samples, each represented by a flattened 10×10 grayscale image (100 features). Because the dataset is sorted by label (all digit-1 samples come first, then digit-2, etc.), the first 3000 samples do **not** contain digits 5 or 6. Thus, the assignment’s suggestion to use “images 1–3000” cannot be applied directly for this binary pair.

To construct balanced and meaningful splits, we extracted **2000 samples of digit 5** and **2000 samples of digit 6** from the full dataset. From each digit, we allocated:

- **1000 samples for training**  
- **1000 samples for validation**  
- **1000 samples for testing**  

This produced class-balanced splits of:

- **Training set:** 2000 samples  
- **Validation set:** 2000 samples  
- **Test set:** 2000 samples  

Just as with logistic regression, we mapped digit **5 → 0** and digit **6 → 1** to form binary labels.

---

### How Random Forests Operate

A Random Forest is an ensemble machine learning algorithm that builds many decision trees and aggregates their predictions through **majority voting**. Each tree is trained on:

1. A random subset of the training data (**bootstrap sampling**), and  
2. A random subset of features at each decision point (`max_features='sqrt'` by default).

These forms of randomness reduce correlation between trees and prevent overfitting, making Random Forests much more robust than individual decision trees. Furthermore, they naturally learn nonlinear relationships and feature interactions, giving them strong classification power even without feature scaling.

---

### Cross-Validation and Hyperparameter Tuning

The main hyperparameter we tuned was **`n_estimators`**, the number of trees in the ensemble. Larger forests tend to perform better but take more time to train, and after a certain point provide diminishing returns.

We evaluated the following set of candidate values using **5-fold cross-validation**:
n_list = [10, 50, 100, 200, 500]


The cross-validation accuracies were:

| Trees | CV Accuracy |
|------|-------------|
| 10   | 0.9710      |
| 50   | 0.9770      |
| 100  | 0.9775      |
| 200  | 0.9800      |
| 500  | **0.9820**  |

The best-performing configuration used **500 trees**, achieving a cross-validation accuracy of **98.20%**.

---

### Final Training and Evaluation

After selecting the optimal number of trees, we retrained the Random Forest on the combined **training + validation** sets (4000 samples total) and evaluated it on the **2000-sample test set**.

The final test accuracy was:

- **97.8%**

The confusion matrix:
[[983 17]
[ 27 973]]


From this we observe:

- 17 digit-5 samples misclassified as digit 6  
- 27 digit-6 samples misclassified as digit 5  
- Only **44 errors** out of **2000** samples  
- Both classes exhibit very similar performance, showing no significant class imbalance or bias  

The classification report:

- Precision: 0.97–0.98  
- Recall: 0.97–0.98  
- F1-score: 0.98 for both classes  

demonstrates that the model performs strongly and symmetrically across both digits.

---

### Summary

The Random Forest classifier produced excellent results on the MNISTmini digit-5-vs-6 binary classification task, achieving **97.8% test accuracy** and outperforming most smaller tree ensembles. Through cross-validation, we determined that a large forest of **500 trees** provided the highest accuracy and generalization performance. Random Forests offer strong nonlinear modeling capacity and robustness, making them a powerful complement to the more linear Logistic Regression approach.

Both models performed well, but Random Forests benefited from their ability to capture more complex feature relationships in the digit data, while Logistic Regression maintained simplicity and interpretability.

---
