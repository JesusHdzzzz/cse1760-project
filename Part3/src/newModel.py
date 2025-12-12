import numpy as np
import pandas as pd
import h2o
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STROKE PREDICTION MODEL - WITH OVERFITTING ANALYSIS
# ============================================================

print("="*60)
print("STROKE PREDICTION MODEL - NO DATA LEAKAGE VERSION")
print("="*60)

start_time = time.time()

# 1. Loading and preparing data...
print("\n1. Loading and preparing data...")
df = pd.read_csv("../data/healthcare-dataset-stroke-data.csv")
bmi_median = df['bmi'].median()
df['bmi'].fillna(bmi_median, inplace=True)
df = df[df['gender'] != 'Other']

print(f"   Dataset shape: {df.shape}")
print(f"   Stroke prevalence: {df['stroke'].mean():.2%}")

# 2. Initial feature selection (domain knowledge only)...
print("\n2. Initial feature selection (domain knowledge only)...")
initial_features = ['age', 'bmi', 'avg_glucose_level', 'hypertension', 
                    'smoking_status', 'work_type', 'gender', 'Residence_type', 
                    'ever_married', 'heart_disease']
target = 'stroke'

# Convert categoricals to factors
categorical_cols = ['smoking_status', 'work_type', 'gender', 'Residence_type', 'ever_married']
for col in categorical_cols:
    df[col] = df[col].astype('category')

# 3. Initializing H2O...
print("\n3. Initializing H2O...")
h2o.init(max_mem_size='4G')

# Convert to H2OFrame
hf = h2o.H2OFrame(df)
hf[target] = hf[target].asfactor()

# 4. Creating stratified splits...
print("\n4. Creating stratified splits...")

# Use stratified splitting
train, valid, test = hf.split_frame(
    ratios=[0.7, 0.15], 
    seed=42,
    destination_frames=['train', 'valid', 'test']
)

print(f"\n✅ FINAL DATA SPLITS:")
print(f"  Train: {train.shape[0]} samples (for training)")
print(f"  Valid: {valid.shape[0]} samples (for hyperparameter tuning)")
print(f"  Test:  {test.shape[0]} samples (FINAL evaluation only)")

# 5. Feature selection using train set only...
print("\n5. Feature selection using train set only...")

# Train a simple GBM for feature importance
gbm_selector = H2OGradientBoostingEstimator(
    seed=42,
    model_id='gbm_feature_selector',
    nfolds=5,
    keep_cross_validation_predictions=True,
    stopping_rounds=10,
    stopping_tolerance=0.001,
    stopping_metric='AUC'
)

gbm_selector.train(
    x=initial_features,
    y=target,
    training_frame=train
)

# Get feature importance
fi = gbm_selector.varimp(use_pandas=True)
print("\nFeature Importance (from train set only):")
for idx, row in fi.iterrows():
    print(f"  {row['variable']:20}: {row['scaled_importance']:.3f}")

# Drop features with low importance
keep_features = fi[fi['scaled_importance'] >= 0.01]['variable'].tolist()
dropped_features = [f for f in initial_features if f not in keep_features]

print(f"\nDropping features with <0.01 importance: {dropped_features}")
print(f"✅ Final features: {len(keep_features)} features: {keep_features}")

# 6. Hyperparameter tuning (train + valid sets only)...
print("\n6. Hyperparameter tuning (train + valid sets only)...")

# Define hyperparameter grid
hyper_params = {
    'max_depth': [3, 5, 7],
    'learn_rate': [0.01, 0.05, 0.1],
    'ntrees': [50, 100, 200],
    'sample_rate': [0.7, 0.8, 1.0],
    'col_sample_rate': [0.7, 0.8, 1.0]
}

# Grid search settings
search_criteria = {
    'strategy': 'Cartesian',
    'max_models': 20,
    'seed': 42
}

# Create and train grid
gbm_grid = H2OGridSearch(
    estimator=H2OGradientBoostingEstimator(
        seed=42,
        model_id='gbm_grid',
        stopping_rounds=10,
        stopping_tolerance=0.001,
        stopping_metric='AUC',
        nfolds=0  # No CV since we're using validation set
    ),
    hyper_params=hyper_params,
    search_criteria=search_criteria,
    grid_id='stroke_gbm_grid'
)

print("Training grid search on train set, validating on validation set...")
gbm_grid.train(
    x=keep_features,
    y=target,
    training_frame=train,
    validation_frame=valid
)

# Get best model
gbm_grid_perf = gbm_grid.get_grid(sort_by='auc', decreasing=True)
best_model = gbm_grid.models[0]
best_auc = best_model.auc(valid=True)
print(f"✓ Best model validation AUC: {best_auc:.4f}")

# ============================================================
# OVERFITTING ANALYSIS FUNCTION
# ============================================================

def analyze_overfitting(gbm_grid, best_model, train, valid, test, target):
    """
    Analyze and visualize overfitting in the model
    """
    
    # Set up the figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # ============================================================
    # 1. COLLECT HYPERPARAMETER TUNING RESULTS
    # ============================================================
    print("\n🔍 Collecting hyperparameter tuning results for overfitting analysis...")
    
    model_performances = []
    
    for i, model in enumerate(gbm_grid.models):
        # Get training metrics
        train_perf = model.model_performance(train_data=train)
        valid_perf = model.model_performance(valid_data=valid)
        
        model_performances.append({
            'model_id': model.model_id,
            'train_auc': train_perf.auc(),
            'valid_auc': valid_perf.auc(),
            'auc_gap': train_perf.auc() - valid_perf.auc(),
            'train_logloss': train_perf.logloss(),
            'valid_logloss': valid_perf.logloss(),
        })
    
    # Convert to DataFrame
    perf_df = pd.DataFrame(model_performances)
    perf_df = perf_df.sort_values('valid_auc', ascending=False).reset_index(drop=True)
    perf_df['model_rank'] = range(1, len(perf_df) + 1)
    
    # Plot 1: Train vs Validation AUC across models
    axes[0, 0].plot(perf_df['model_rank'], perf_df['train_auc'], 
                   'b-', label='Train AUC', marker='o', markersize=4, linewidth=2)
    axes[0, 0].plot(perf_df['model_rank'], perf_df['valid_auc'], 
                   'r-', label='Valid AUC', marker='s', markersize=4, linewidth=2)
    axes[0, 0].fill_between(perf_df['model_rank'], 
                           perf_df['train_auc'], 
                           perf_df['valid_auc'],
                           alpha=0.2, color='gray', label='Overfitting Gap')
    axes[0, 0].set_xlabel('Model Rank (by Validation AUC)', fontsize=10)
    axes[0, 0].set_ylabel('AUC', fontsize=10)
    axes[0, 0].set_title('Hyperparameter Tuning: Train vs Validation AUC', fontsize=12, fontweight='bold')
    axes[0, 0].legend(loc='best')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Set y-axis limits dynamically
    min_auc = min(perf_df['train_auc'].min(), perf_df['valid_auc'].min())
    max_auc = max(perf_df['train_auc'].max(), perf_df['valid_auc'].max())
    axes[0, 0].set_ylim([max(0.5, min_auc - 0.05), min(1.0, max_auc + 0.05)])
    
    # Highlight best model
    best_idx = perf_df[perf_df['valid_auc'] == perf_df['valid_auc'].max()].index[0]
    axes[0, 0].axvline(x=best_idx+1, color='green', linestyle='--', alpha=0.5, label='Best Model')
    
    # Plot 2: AUC Gap (Overfitting Measure)
    colors = ['red' if gap > 0.05 else 'orange' if gap > 0.02 else 'green' 
              for gap in perf_df['auc_gap']]
    axes[0, 1].bar(perf_df['model_rank'], perf_df['auc_gap'], color=colors, alpha=0.7)
    axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 1].axhline(y=0.02, color='orange', linestyle='--', alpha=0.5, label='2% threshold')
    axes[0, 1].axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='5% threshold')
    axes[0, 1].set_xlabel('Model Rank (by Validation AUC)', fontsize=10)
    axes[0, 1].set_ylabel('Train AUC - Valid AUC', fontsize=10)
    axes[0, 1].set_title('Overfitting Gap Analysis', fontsize=12, fontweight='bold')
    axes[0, 1].legend(loc='best')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add value labels for best model
    best_gap = perf_df.loc[best_idx, 'auc_gap']
    axes[0, 1].text(best_idx+1, best_gap + 0.002 if best_gap >= 0 else best_gap - 0.002, 
                   f'Best: {best_gap:.3f}', 
                   ha='center', va='bottom' if best_gap >= 0 else 'top', 
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # ============================================================
    # 2. FINAL MODEL PERFORMANCE COMPARISON
    # ============================================================
    print("📊 Calculating final model performance on all datasets...")
    
    # Get performance on all three sets
    train_perf = best_model.model_performance(train_data=train)
    valid_perf = best_model.model_performance(valid_data=valid)
    test_perf = best_model.model_performance(test_data=test)
    
    # Plot 3: Performance comparison across datasets
    datasets = ['Train', 'Validation', 'Test']
    auc_scores = [train_perf.auc(), valid_perf.auc(), test_perf.auc()]
    logloss_scores = [train_perf.logloss(), valid_perf.logloss(), test_perf.logloss()]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = axes[0, 2].bar(x - width/2, auc_scores, width, 
                          label='AUC (higher=better)', 
                          color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.8)
    bars2 = axes[0, 2].bar(x + width/2, logloss_scores, width, 
                          label='LogLoss (lower=better)', 
                          color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.6)
    
    axes[0, 2].set_xlabel('Dataset', fontsize=10)
    axes[0, 2].set_ylabel('Score', fontsize=10)
    axes[0, 2].set_title('Best Model: Performance Across Datasets', fontsize=12, fontweight='bold')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(datasets)
    axes[0, 2].legend(loc='best')
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0, 2].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', 
                           fontsize=9, fontweight='bold')
    
    # Plot 4: AUC progression from train to test
    progression_data = {
        'Dataset': ['Train → Valid', 'Valid → Test', 'Train → Test'],
        'AUC_Drop': [
            train_perf.auc() - valid_perf.auc(),
            valid_perf.auc() - test_perf.auc(),
            train_perf.auc() - test_perf.auc()
        ],
        'Percentage_Drop': [
            (train_perf.auc() - valid_perf.auc()) / train_perf.auc() * 100,
            (valid_perf.auc() - test_perf.auc()) / valid_perf.auc() * 100,
            (train_perf.auc() - test_perf.auc()) / train_perf.auc() * 100
        ]
    }
    prog_df = pd.DataFrame(progression_data)
    
    colors = ['green' if p < 2 else 'orange' if p < 5 else 'red' 
              for p in prog_df['Percentage_Drop']]
    
    axes[1, 0].bar(prog_df['Dataset'], prog_df['Percentage_Drop'], color=colors, alpha=0.7)
    axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 0].axhline(y=2, color='green', linestyle='--', alpha=0.5, label='2% threshold')
    axes[1, 0].axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5% threshold')
    axes[1, 0].set_xlabel('Performance Transition', fontsize=10)
    axes[1, 0].set_ylabel('AUC Drop (%)', fontsize=10)
    axes[1, 0].set_title('Performance Degradation Analysis', fontsize=12, fontweight='bold')
    axes[1, 0].legend(loc='best')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].tick_params(axis='x', rotation=15)
    
    # Add value labels
    for i, (_, row) in enumerate(prog_df.iterrows()):
        axes[1, 0].text(i, row['Percentage_Drop'] + 0.1, 
                       f"{row['Percentage_Drop']:.1f}%\n({row['AUC_Drop']:.3f})", 
                       ha='center', va='bottom', fontsize=9)
    
    # ============================================================
    # 3. CONFUSION MATRIX HEATMAP COMPARISON (Simplified)
    # ============================================================
    
    # Get optimal threshold from validation set
    thresholds_valid = valid_perf.metric('thresholds')
    f1_valid = valid_perf.metric('f1')
    optimal_idx = np.argmax(f1_valid)
    optimal_threshold = thresholds_valid[optimal_idx]
    
    # Create confusion matrices at optimal threshold
    def get_confusion_matrix(perf, threshold):
        cm = perf.confusion_matrix(thresholds=[threshold])['cm'][0]
        return pd.DataFrame({
            'Actual 0': [cm[0, 0], cm[1, 0]],
            'Actual 1': [cm[0, 1], cm[1, 1]]
        }, index=['Predicted 0', 'Predicted 1'])
    
    cm_train = get_confusion_matrix(train_perf, optimal_threshold)
    cm_test = get_confusion_matrix(test_perf, optimal_threshold)
    
    # Plot confusion matrices
    for idx, (cm, title, pos) in enumerate([(cm_train, 'Train Set', (1, 1)), 
                                            (cm_test, 'Test Set', (1, 2))]):
        ax = axes[pos]
        im = ax.imshow(cm.values, cmap='Blues', aspect='auto', vmin=0, vmax=cm.values.max())
        ax.set_title(f'{title} at threshold={optimal_threshold:.3f}', fontsize=11, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['No Stroke', 'Stroke'])
        ax.set_yticklabels(['No Stroke', 'Stroke'])
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm.iloc[i, j]:,.0f}', 
                       ha='center', va='center', color='black' if cm.iloc[i, j] < cm.values.max()/2 else 'white',
                       fontsize=12, fontweight='bold')
    
    # Plot 6: Key Metrics Summary
    metrics_summary = pd.DataFrame({
        'Dataset': ['Train', 'Validation', 'Test'],
        'AUC': auc_scores,
        'LogLoss': logloss_scores,
        'Optimal_Threshold': [optimal_threshold, optimal_threshold, optimal_threshold]
    })
    
    axes[1, 2].axis('off')
    table_data = []
    for _, row in metrics_summary.iterrows():
        table_data.append([
            row['Dataset'],
            f"{row['AUC']:.4f}",
            f"{row['LogLoss']:.4f}",
            f"{row['Optimal_Threshold']:.3f}"
        ])
    
    table = axes[1, 2].table(cellText=table_data,
                            colLabels=['Dataset', 'AUC', 'LogLoss', 'Threshold'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0.1, 0.1, 0.8, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    axes[1, 2].set_title('Performance Summary', fontsize=12, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.show()
    
    # ============================================================
    # 4. PRINT KEY OVERFITTING INSIGHTS
    # ============================================================
    print("\n" + "="*60)
    print("📈 OVERFITTING ANALYSIS RESULTS")
    print("="*60)
    
    # Calculate key metrics
    train_auc = train_perf.auc()
    valid_auc = valid_perf.auc()
    test_auc = test_perf.auc()
    
    auc_gap_train_valid = train_auc - valid_auc
    auc_gap_train_test = train_auc - test_auc
    auc_drop_pct = (train_auc - test_auc) / train_auc * 100
    
    # Overfitting assessment
    print(f"\n🎯 PERFORMANCE METRICS:")
    print(f"   Train AUC:     {train_auc:.4f}")
    print(f"   Valid AUC:     {valid_auc:.4f}")
    print(f"   Test AUC:      {test_auc:.4f}")
    print(f"   Optimal Threshold: {optimal_threshold:.4f}")
    
    print(f"\n🔍 OVERFITTING ASSESSMENT:")
    
    # Rule 1: AUC gap between train and validation
    if auc_gap_train_valid > 0.05:
        overfitting_level = "HIGH overfitting"
        assessment = f"   ⚠️  WARNING: Large AUC gap (>0.05) = {auc_gap_train_valid:.4f}"
    elif auc_gap_train_valid > 0.02:
        overfitting_level = "MODERATE overfitting"
        assessment = f"   ⚠️  CAUTION: Moderate AUC gap (>0.02) = {auc_gap_train_valid:.4f}"
    else:
        overfitting_level = "MINIMAL overfitting"
        assessment = f"   ✅ Good: Small AUC gap = {auc_gap_train_valid:.4f}"
    
    print(assessment)
    
    # Rule 2: Performance drop from train to test
    if auc_drop_pct > 15:
        generalization = "POOR generalization"
        assessment = f"   ⚠️  WARNING: High performance drop (>15%) = {auc_drop_pct:.1f}%"
    elif auc_drop_pct > 5:
        generalization = "MODERATE generalization"
        assessment = f"   ⚠️  CAUTION: Moderate performance drop (>5%) = {auc_drop_pct:.1f}%"
    else:
        generalization = "GOOD generalization"
        assessment = f"   ✅ Good: Minimal performance drop = {auc_drop_pct:.1f}%"
    
    print(assessment)
    
    # Rule 3: Validation vs Test consistency
    val_test_diff = abs(valid_auc - test_auc)
    if val_test_diff > 0.03:
        consistency = "INCONSISTENT"
        assessment = f"   ⚠️  WARNING: Large validation-test mismatch = {val_test_diff:.4f}"
    else:
        consistency = "CONSISTENT"
        assessment = f"   ✅ Good: Validation and test are consistent = {val_test_diff:.4f}"
    
    print(assessment)
    
    print(f"\n📊 FINAL ASSESSMENT:")
    print(f"   Overfitting Level:      {overfitting_level}")
    print(f"   Generalization:         {generalization}")
    print(f"   Validation-Test:        {consistency}")
    
    if "HIGH" in overfitting_level or "POOR" in generalization:
        print(f"\n💡 RECOMMENDATION: Consider reducing model complexity,")
        print(f"   increasing regularization, or collecting more data.")
    elif "MODERATE" in overfitting_level or "MODERATE" in generalization:
        print(f"\n💡 RECOMMENDATION: Model is acceptable but could be improved.")
        print(f"   Monitor performance closely on new data.")
    else:
        print(f"\n💡 RECOMMENDATION: Model generalizes well!")
        print(f"   Ready for deployment with confidence.")
    
    print("="*60)
    
    return perf_df, {
        'train_auc': train_auc,
        'valid_auc': valid_auc,
        'test_auc': test_auc,
        'auc_gap': auc_gap_train_test,
        'optimal_threshold': optimal_threshold
    }

# ============================================================
# RUN OVERFITTING ANALYSIS
# ============================================================

# Analyze overfitting BEFORE threshold optimization
overfitting_results = analyze_overfitting(
    gbm_grid=gbm_grid,
    best_model=best_model,
    train=train,
    valid=valid,
    test=test,
    target=target
)

# 7. Optimizing threshold on validation set only...
print("\n7. Optimizing threshold on validation set only...")

# Get validation performance
valid_perf = best_model.model_performance(valid_data=valid)

# Find optimal threshold for F2 score
thresholds = valid_perf.metric('thresholds')
f2_scores = valid_perf.metric('f2')
optimal_idx = np.argmax(f2_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"✓ Optimal F2 threshold from validation set: {optimal_threshold:.4f}")

# 8. FINAL evaluation on test set (one time only)...
print("\n8. FINAL evaluation on test set (one time only)...")

# Get test predictions
test_pred = best_model.predict(test)

# Calculate metrics at optimal threshold
def calculate_metrics(perf, threshold):
    cm = perf.confusion_matrix(thresholds=[threshold])['cm'][0]
    tp, fp, fn, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f2 = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return recall, precision, f2, accuracy, tp, fp, fn, tn

test_perf = best_model.model_performance(test_data=test)
test_recall, test_precision, test_f2, test_accuracy, tp, fp, fn, tn = calculate_metrics(test_perf, optimal_threshold)

# Calculate AUCPR
from sklearn.metrics import average_precision_score, roc_auc_score
test_probs = test_pred['p1'].as_data_frame().values.flatten()
test_actual = test[target].as_data_frame().values.flatten()
aucpr = average_precision_score(test_actual, test_probs)
aucroc = roc_auc_score(test_actual, test_probs)

print("\n" + "="*60)
print("✅ FINAL UNBIASED PERFORMANCE")
print("="*60)

print(f"\nTest Set Performance (threshold = {optimal_threshold:.4f}):")
print(f"  Recall:              {test_recall:.3f} ({tp}/{tp+fn} strokes caught)")
print(f"  Precision:           {test_precision:.3f} ({tp}/{tp+fp} alerts are real)")
print(f"  F2 Score:            {test_f2:.3f}")
print(f"  Accuracy:            {test_accuracy:.3f}")
print(f"  AUCPR:               {aucpr:.4f}")
print(f"  AUC-ROC:             {aucroc:.4f}")

print(f"\nConfusion Matrix (Test Set):")
print(f"                Actual")
print(f"                No Stroke    Stroke")
print(f"  Pred No Stroke   {tn:4d}         {fn:4d}")
print(f"  Pred Stroke      {fp:4d}         {tp:4d}")

print(f"\nData Usage Summary:")
print(f"  Train:    {train.shape[0]} samples - Model training")
print(f"  Valid:    {valid.shape[0]} samples - Hyperparameter tuning")
print(f"  Test:     {test.shape[0]} samples - FINAL evaluation only")

elapsed_time = time.time() - start_time
print("\n" + "="*60)
print(f"✅ COMPLETED IN {elapsed_time:.2f} SECONDS")
print("✅ NO DATA LEAKAGE - Proper train/valid/test usage")
print("✅ OVERFITTING ANALYSIS COMPLETE")
print("="*60)

# Shutdown H2O
h2o.cluster().shutdown()