# Curated Portfolio Results

## Overview

This directory is a compact, reviewer-facing snapshot of the latest successful CSE 176 portfolio rerun. It contains selected metrics and validated visuals, not the full generated output. Bulk artifacts remain under the Git-ignored `part1/outputs/`, `part2/outputs/`, and `part3/outputs/` trees.

| Part | Experiment | Selection metric | Final test metric(s) | Key result |
|---|---|---|---|---|
| 1 | Logistic regression | 5-fold CV accuracy: 0.9700 | Accuracy: 0.9740 | Strong linear baseline; selected `C=0.1`. |
| 1 | Random forest | 5-fold CV accuracy: 0.9770 | Accuracy: 0.9820 | Modestly outperformed logistic regression; selected 500 trees. |
| 2 | XGBoost on pixels | 2-fold CV accuracy: 0.9441; validation error: 0.0398 | Accuracy: 0.9610 | Raw-pixel baseline; validation selected 200 trees. |
| 2 | XGBoost on LeNet features | 3-fold CV accuracy: 0.9803; validation error: 0.0174 | Accuracy: 0.9833 | Precomputed LeNet features improved downstream XGBoost accuracy by 2.23 percentage points. |
| 3 | Stroke random forest | 5-fold CV average precision: 0.1693 | AP: 0.2230; ROC AUC: 0.8233 | Competitive imbalanced-classification baseline. |
| 3 | Stroke H2O GBM | 5-fold CV AUCPR: 0.2215 | AP: 0.2329; ROC AUC: 0.8396 | Modest improvement over the random forest. |
| 3 | SuperCon GBM | Validation RMSE: 10.1575 | RMSE: 9.4400; MAE: 5.2474 | Best composition-disjoint configuration: `gbm_n1100_d11_lr03`. |
| 3 | SuperCon target-conditioned diagnostic | Not applicable | Diagnostic RMSE — low: 1.5522; medium: 5.5690; high: 11.1098 | Diagnostic only: true target values route rows to buckets, so this is non-deployable and is not primary model performance. |

## Interpretation

**Part 1 — binary MNIST classification.** Random forest test accuracy was 0.8 percentage points higher than logistic regression. Both test metrics come from final models refit on the combined training and validation rows after cross-validation-based selection.

**Part 2 — representation comparison.** With the reported search procedures, XGBoost on precomputed LeNet features reached 0.9833 accuracy versus 0.9610 on pixels. The comparison supports the value of the learned representation, but the provenance of the LeNet extractor/checkpoint is incomplete, so this result should not be presented as fully reproducible representation learning.

**Part 3 — stroke classification.** Average precision is emphasized because only 50 of 1,022 test rows are positive. H2O GBM improved test average precision by about 0.010 and ROC AUC by about 0.016 over random forest—a modest, not dramatic, advantage.

**Part 3 — SuperCon regression.** The primary GBM uses a composition-disjoint split based on normalized elemental compositions. This tests separation by composition groups; it does not establish performance on "completely unseen materials." Elemental composition omits crystal structure, pressure, phase, synthesis conditions, defects, and other factors that can affect critical temperature.

**Target-conditioned diagnostic.** The low/medium/high bucket models use the true critical temperature to choose a bucket. Their metrics are useful for error analysis only, are non-deployable, and must not be compared as primary end-to-end SuperCon performance.

## Provenance

- Source run: `portfolio-2026-09-09_19-22-42`
- Source commit: `fb9cf477da00c505d6403258ea2f0cd622572787`
- Python: 3.13.11
- Random seed: 42 for all reported experiments
- Part 2 pixels data SHA-256: `b22e20253d929c06f2a925f0e90596a908c2b8cb73346428eec57364d6ecfb2f`
- Part 2 LeNet-feature data SHA-256: `a1ad9e80de3a473a451ac6c3d32b7c1111b83fd9450e648616cfafe5436674a2`
- Other input hashes were not recorded in the generated metadata.

The source run's summary marked every portfolio job as passed. Metrics in this directory were transcribed from that run's JSON/CSV metadata; selected plots and supporting CSVs are byte-for-byte copies with clearer filenames. No experiment was rerun for this curation.

## Contents

- `part1/`: concise metrics and one model-selection plot per classifier
- `part2/`: concise metrics and one validation curve per representation
- `part3/stroke/`: test precision-recall and ROC curves for the strongest stroke classifier
- `part3/supercon/`: primary validation/model-selection results and target-distribution context
- `part3/diagnostics/`: target-conditioned bucket metrics, explicitly separated from deployable results
