# Methodology — Mobile Market Intelligence Platform

---

## 1. Dataset Preprocessing

### Loading
- Loaded with `pd.read_csv(..., encoding='utf-8', encoding_errors='replace')` to handle encoding artifacts
- Trailing blank row dropped with `dropna(how='all')`

### Brand Normalisation
- `Poco` → `POCO` (capitalisation standardised)

### Deduplication
- Composite key: `Company Name + Model Name`
- Duplicate rows removed using `df.duplicated(subset=..., keep='first')`
- ~26 exact duplicates removed (Oppo K-series repeated blocks)

---

## 2. Feature Engineering

All raw columns preserved. New analytical columns added:

| Column | Transformation |
|---|---|
| RAM_num | `re.findall(r"\d+\.?\d*", val)[0]` → float |
| Weight_num | Extract first number from `174g` |
| FrontCam_num | Extract first MP value; normalise `Dual 32MP → 32` |
| BackCam_primary | Extract first MP value |
| BackCam_lenses | Count of ` + ` separators + 1 |
| Battery_num | Remove all non-digits from `3,600mAh` |
| Screen_num | Extract first float from `6.7 inches (main), 2.7 inches (external)` |
| India_Price | Strip `INR ` + all commas; handles `1,04,999` → `104999` |
| Pakistan_Price | Strip `PKR `; `Not available` → NaN |
| China_Price | Strip `CNY `; encoding artifacts → NaN |
| USA_Price | Strip `USD `; `396,22` → `396.22` (European decimal) |
| Dubai_Price | Strip `AED ` + commas |

---

## 3. KPI Calculations

All KPIs are computed from the cleaned dataset using Pandas aggregations:

- **Total Smartphones**: `len(df)` after cleaning
- **Unique Brands**: `df['Company Name'].nunique()`
- **Avg India Price**: `df['India_Price'].mean()`
- **Median India Price**: `df['India_Price'].median()`
- **Avg RAM**: `df['RAM_num'].mean()`
- **Avg Battery**: `df['Battery_num'].mean()`
- **Avg Screen**: `df['Screen_num'].mean()`
- **Segment Distribution**: `df['Price_Segment'].value_counts()`

---

## 4. Price Segmentation

Thresholds are fixed values chosen to approximate dataset percentiles (P20/40/65/85):

| Segment | Range |
|---|---|
| Budget | < ₹15,000 |
| Mid-Range | ₹15,000 – ₹30,000 |
| Upper Mid-Range | ₹30,000 – ₹60,000 |
| Premium | ₹60,000 – ₹1,00,000 |
| Ultra Premium | > ₹1,00,000 |

These thresholds are documented in `DATA_DICTIONARY.md` and displayed in the UI.

---

## 5. Statistical Analysis

### Correlation
- Pearson product-moment correlation via `Series.corr()`
- Computed between India_Price and: RAM, Battery, Screen, Front Camera, Back Camera, Weight, Year
- All correlations presented with the disclaimer: "correlation ≠ causation"

### Correlation Matrix
- Full pairwise correlation matrix using `DataFrame.corr()`
- 8 numeric features: India_Price, RAM_num, Battery_num, Screen_num, FrontCam_num, BackCam_primary, Weight_num, Launched Year

---

## 6. Visualization Methodology

- All charts use only dataset-derived values — no estimated or placeholder data
- Scatter plots sample up to 500 points for performance; correlation is computed on full dataset
- Recharts library used for all visualizations
- Color scales use consistent palette across all charts

---

## 7. Machine Learning

### Target Variable
`India_Price` (Indian Rupee launch price)

### Feature Set
Numeric: `RAM_num, Battery_num, FrontCam_num, BackCam_primary, BackCam_lenses, Screen_num, Weight_num, Launched Year`  
Categorical: `Company Name, Processor_Family` (one-hot encoded via `OneHotEncoder(handle_unknown='ignore')`)

### Exclusions (Anti-Leakage)
- `Model Name` — not a causal predictor; would cause memorisation
- `Pakistan_Price, China_Price, USA_Price, Dubai_Price` — would directly encode the target
- `Price_Segment` — derived from `India_Price` itself

### Preprocessing Pipeline
```
ColumnTransformer
├── passthrough → numeric features
└── OneHotEncoder(handle_unknown='ignore') → categorical features
→ Pipeline(preprocessor, estimator)
```

### Train/Test Split
- 80% training / 20% test
- `random_state=42` for reproducibility
- No cross-validation used (could improve estimates but adds complexity)

### Models

| Model | Parameters |
|---|---|
| Baseline | Always predicts `y_train.mean()` |
| Linear Regression | Default sklearn |
| Decision Tree | `max_depth=8, random_state=42` |
| Random Forest | `n_estimators=200, max_depth=12, random_state=42, n_jobs=-1` |
| Gradient Boosting | `n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42` |

### Evaluation Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| MAE | mean(|y_true - y_pred|) | Average ₹ error magnitude |
| RMSE | sqrt(mean((y_true - y_pred)²)) | Penalises large errors more |
| R² | 1 - SS_res / SS_tot | Fraction of variance explained; 1.0 = perfect |

### Prediction Uncertainty
- Approximated as ±1 standard deviation of residuals on the test set
- Not a true confidence interval (would require quantile regression or calibration)

---

## 8. Automatic Insights

All insights are generated programmatically from the cleaned dataset:

1. Top brand by model count (`value_counts().idxmax()`)
2. Median India price (`median()`)
3. Most common RAM (`mode()[0]`)
4. Price trend 2020 → latest year (mean comparison)
5. Battery-price Pearson correlation
6. RAM-price Pearson correlation
7. Most common processor family
8. Budget vs Ultra Premium % split
9. Most active launch year
10. Screen size-price correlation

No insight is hardcoded — all values recalculate if the dataset changes.

---

## 9. Data Quality Score

100-point transparent scoring system:

| Rule | Points |
|---|---|
| No missing India prices | 20 |
| No duplicate records | 20 |
| No encoding errors in China prices | 15 |
| All numeric fields parseable | 20 |
| Launch year within valid range (2014–2025) | 10 |
| No invalid spec values | 15 |

---

## 10. Limitations and Caveats

- **Tablet inclusion**: Dataset includes iPads, Samsung Galaxy Tabs, etc. These have different price/spec profiles. Filter with `Is_Tablet = True/False`.
- **Camera simplification**: Back camera strings like `50MP (Main) + 50MP (Ultra-wide) + 50MP (Telephoto)` are simplified to the first value (50 MP). Multi-camera complexity is partially captured by `BackCam_lenses`.
- **Foldable screens**: Strings like `"6.7 inches (main), 2.7 inches (external)"` are reduced to the main screen value (6.7").
- **Processor granularity**: 200+ processor strings are grouped into ~15 families. Some nuance is lost (e.g., Snapdragon 865 vs 888 are in the same family).
- **Price vintage**: Launch prices may not reflect current retail prices.
- **Model generalisation**: ML models trained on this specific dataset. Predictions outside this distribution may be unreliable.
- **No causal claims**: All correlations are observational, not causal.
