# Model Card — Mobile Market Intelligence Price Prediction Models

---

## Overview

This model card documents the five regression models used to estimate India smartphone launch prices from device specifications.

**Date trained:** At application startup (on-demand)  
**Dataset:** Mobiles Dataset (2025) — cleaned records only  
**Task:** Supervised regression — predict India launch price (₹)  
**Intended use:** Educational and exploratory data analysis; not for commercial pricing decisions

---

## Models

### 1. Baseline (Mean)

| Field | Value |
|---|---|
| Objective | Always predicts the training set mean — establishes lower performance bound |
| Target | India_Price |
| Features | None (no features used) |
| Training method | Compute `y_train.mean()` |
| Limitations | Cannot account for any device characteristics |
| Bias risk | None from features; inherits any distribution bias from the dataset |

---

### 2. Linear Regression

| Field | Value |
|---|---|
| Objective | Estimate price as a linear combination of weighted features |
| Target | India_Price |
| Features | All 8 numeric + one-hot encoded Company + Processor Family |
| Training method | Ordinary Least Squares via scikit-learn default |
| Limitations | Cannot model non-linear relationships; sensitive to outliers |
| Bias risk | One-hot encoding assumes brand/processor effects are independent additive constants |

---

### 3. Decision Tree Regressor

| Field | Value |
|---|---|
| Objective | Hierarchical rule-based price splitting |
| Target | India_Price |
| Features | All 8 numeric + one-hot encoded Company + Processor Family |
| Parameters | `max_depth=8, random_state=42` |
| Limitations | Prone to overfitting; max_depth limits complexity |
| Bias risk | Tree splits may amplify imbalanced brand representation |

---

### 4. Random Forest Regressor

| Field | Value |
|---|---|
| Objective | Ensemble of decision trees for robust price estimation |
| Target | India_Price |
| Features | All 8 numeric + one-hot encoded Company + Processor Family |
| Parameters | `n_estimators=200, max_depth=12, random_state=42, n_jobs=-1` |
| Training method | Bagging with random feature subsets |
| Limitations | Less interpretable than linear models; larger memory footprint |
| Bias risk | Over-represented brands (Samsung, Oppo, Honor) may dominate feature importance |
| Feature importance | Built-in Gini impurity-based importance |

---

### 5. Gradient Boosting Regressor

| Field | Value |
|---|---|
| Objective | Sequential boosting for high-accuracy price estimation |
| Target | India_Price |
| Features | All 8 numeric + one-hot encoded Company + Processor Family |
| Parameters | `n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42` |
| Training method | Gradient descent on squared loss |
| Limitations | Slower training; susceptible to outliers in high-price segment |
| Bias risk | Same brand/processor representation concerns as Random Forest |

---

## Evaluation Methodology

### Data Split
- 80% training / 20% test
- `random_state=42`
- No stratification (price is continuous)

### Metrics

| Metric | Formula | Good Value |
|---|---|---|
| MAE | mean(|y_true - y_pred|) | Lower is better (₹ units) |
| RMSE | √mean((y_true - y_pred)²) | Lower is better (₹ units) |
| R² | 1 - SS_res/SS_tot | Closer to 1.0 is better |

### Anti-Leakage Measures
- `Model Name` excluded (memorisation risk)
- All other price columns excluded
- `Price_Segment` excluded (derived from India_Price)

---

## Uncertainty Quantification

Prediction uncertainty is approximated as ±1 standard deviation of residuals on the test set. This is a rough heuristic, not a calibrated prediction interval. For true uncertainty quantification, consider:
- Quantile regression
- Conformal prediction
- Bayesian approaches

---

## Limitations

1. Models trained on one dataset snapshot — may not generalise to other time periods
2. Camera features simplified (primary MP only) — multi-camera system quality not captured
3. Foldable and tablet devices included — their atypical spec profiles may affect predictions for normal phones
4. Brand effects captured via one-hot encoding — new brands would be encoded as `Unknown`
5. Prediction uncertainty is approximate, not statistically calibrated

---

## Responsible Interpretation

- All predictions come with a mandatory disclaimer: *"This is a machine-learning estimate based on patterns in the provided dataset and should not be treated as an actual market quotation."*
- Feature importance shows statistical association, not causal contribution
- Correlation ≠ causation — high RAM importance does not mean RAM causes high prices
- Do not use predictions for commercial pricing decisions without additional validation

---

## Potential Sources of Bias

| Bias Type | Description |
|---|---|
| Brand representation | Some brands have far more models in the dataset than others, potentially skewing brand coefficients |
| Temporal skew | 2024 has the most models; models may over-represent recent pricing |
| Segment imbalance | Budget devices may outnumber Ultra Premium, affecting tail predictions |
| Tablet inclusion | Tablets have different spec-price relationships; including them may degrade phone-specific predictions |
| Camera simplification | Primary MP only — misses sensor size, aperture, video capabilities |
