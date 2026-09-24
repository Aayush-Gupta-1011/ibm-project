# API Documentation — Mobile Market Intelligence

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

---

## Health

### GET /health
Returns API status.

**Response:**
```json
{ "status": "ok", "service": "Mobile Market Intelligence API" }
```

---

## Overview / KPIs

### GET /api/kpis
Returns all dashboard KPIs.

**Query params:**
- `include_tablets` (bool, default: true)

**Response:**
```json
{
  "total_smartphones": 905,
  "unique_brands": 25,
  "unique_processors": 190,
  "avg_india_price": 45231.50,
  "median_india_price": 29999.00,
  ...
}
```

---

## Market Analysis

### GET /api/market/brand-distribution
Returns model count and market share per brand.

### GET /api/market/yearly-launches
Returns launch count and avg India price per year.

### GET /api/market/ram-distribution
Returns count of models per RAM size.

### GET /api/market/ram-by-year
Returns avg RAM per year.

### GET /api/market/battery-distribution
Returns count of models per battery range.

### GET /api/market/battery-by-brand
Returns avg battery per brand.

### GET /api/market/brand-by-year
Returns per-brand count per year.

---

## Brand Intelligence

### GET /api/brands
Returns list of all brand names.

### GET /api/brand/{brand_name}
Returns full brand profile.

**Example:** `GET /api/brand/Apple`

### GET /api/brands/compare
Returns comparison data for multiple brands.

**Query params:**
- `brands` (string): comma-separated brand names — e.g. `Apple,Samsung,OnePlus`

---

## Price Analysis

### GET /api/price/distribution
Returns price histogram data.

**Query params:** `bins` (int, default: 20)

### GET /api/price/by-segment
Returns per-segment stats.

### GET /api/price/vs-spec/{spec}
Returns scatter data and Pearson correlation.

**spec values:** `ram` | `battery` | `screen` | `front_cam` | `back_cam` | `weight`

### GET /api/price/country-averages
Returns avg and median price per country.

---

## Global Price Comparison

### GET /api/global/models
Returns list of all model names.

### GET /api/global/model
Returns price data for a specific model.

**Query params:** `model` (string)

---

## Specification Explorer

### GET /api/specs/explore
Returns filtered, sorted device records.

**Query params:**
| Param | Type | Description |
|---|---|---|
| brand | string | Filter by brand |
| min_ram / max_ram | float | RAM range (GB) |
| min_battery / max_battery | int | Battery range (mAh) |
| min_price / max_price | float | India price range (₹) |
| min_year / max_year | int | Year range |
| price_segment | string | Budget / Mid-Range / Upper Mid-Range / Premium / Ultra Premium |
| search | string | Text search on model/brand/processor |
| sort_by | string | India_Price / RAM_num / Battery_num / Screen_num / Launched Year |
| sort_asc | bool | Sort direction |
| limit | int | Max results (default 200) |

---

## Smartphone Comparator

### GET /api/compare
Compare 2–4 devices.

**Query params:** `models` (string): comma-separated model names

---

## Analytics

### GET /api/analytics/correlation-matrix
Returns Pearson correlation matrix for numeric features.

---

## Data Quality

### GET /api/data-quality
Returns quality report with score, missing values, and cleaning summary.

---

## Insights

### GET /api/insights
Returns list of auto-generated data insights.

**Response item:**
```json
{
  "category": "Market",
  "title": "Top Brand by Volume",
  "text": "Honor has the most models in this dataset (92 models, 10.2% of total).",
  "type": "info"
}
```

---

## Machine Learning

### GET /api/ml/metrics
Returns evaluation metrics for all trained models.

**Response:**
```json
[
  { "model": "Baseline (Mean)", "mae": 34210.5, "rmse": 44120.2, "r2": 0.0 },
  { "model": "Random Forest", "mae": 12450.0, "rmse": 22310.0, "r2": 0.72 }
]
```

### GET /api/ml/feature-importance
Returns feature importance for a trained model.

**Query params:** `model_name` (string, default: `Random Forest`)

### GET /api/ml/diagnostics
Returns actual vs predicted data and residuals.

**Query params:** `model_name` (string, default: `Random Forest`)

### POST /api/ml/predict
Estimates India launch price from specifications.

**Request body:**
```json
{
  "company": "Samsung",
  "ram_gb": 8.0,
  "battery_mah": 5000,
  "front_cam_mp": 12.0,
  "back_cam_mp": 50.0,
  "back_cam_lenses": 2,
  "screen_inches": 6.5,
  "weight_g": 190.0,
  "launched_year": 2024,
  "processor_family": "Snapdragon 8 (Flagship)",
  "model_name": "Random Forest"
}
```

**Response:**
```json
{
  "estimated_india_price": 84321.50,
  "uncertainty_low": 62100.0,
  "uncertainty_high": 106543.0,
  "model_used": "Random Forest",
  "model_metrics": { "mae": 12450.0, "rmse": 22310.0, "r2": 0.72 },
  "disclaimer": "This is a machine-learning estimate..."
}
```

---

## Processor Families

### GET /api/processor-families
Returns list of all processor family strings (for prediction form dropdowns).

---

## Error Responses

| Status | Meaning |
|---|---|
| 404 | Brand or model not found |
| 400 | Invalid input to prediction endpoint |
| 500 | Server error (logged) |

**Error body:**
```json
{ "detail": "Brand 'XYZ' not found" }
```
