"""
ML training, evaluation and prediction engine.
Models: Baseline (mean), Linear Regression, Decision Tree, Random Forest, Gradient Boosting.
Target: India_Price
No target leakage — model names and raw price columns are excluded from features.
"""

import math
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

from app.preprocessing.cleaner import get_clean_df

CACHE_DIR = Path(__file__).parent / "_cache"
METRICS_FILE = CACHE_DIR / "model_metrics.json"

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "RAM_num", "Battery_num", "FrontCam_num", "BackCam_primary",
    "BackCam_lenses", "Screen_num", "Weight_num", "Launched Year",
]
CATEGORICAL_FEATURES = ["Company Name", "Processor_Family"]
TARGET = "India_Price"


def _build_feature_matrix(df: pd.DataFrame):
    """Extract features and target, drop rows with missing target or features."""
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    sub = df[cols].dropna(subset=[TARGET] + NUMERIC_FEATURES)
    # Fill categorical NaN with "Unknown"
    for c in CATEGORICAL_FEATURES:
        sub[c] = sub[c].fillna("Unknown")
    X = sub[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = sub[TARGET].astype(float)
    return X, y


def _make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def _metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
    }


# ---------------------------------------------------------------------------
# Train all models
# ---------------------------------------------------------------------------

_trained_models: dict = {}
_model_metrics: list = []
_X_test = None
_y_test = None
_feature_names: list = []


def train_all_models(force: bool = False) -> list[dict]:
    global _trained_models, _model_metrics, _X_test, _y_test, _feature_names

    if _trained_models and not force:
        return _model_metrics

    df = get_clean_df()
    X, y = _build_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    _X_test = X_test
    _y_test = y_test

    preprocessor = _make_preprocessor()

    models_def = {
        "Baseline (Mean)": None,  # special case
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
    }

    results = []
    for name, estimator in models_def.items():
        if estimator is None:
            # Baseline: always predict training mean
            train_mean = float(y_train.mean())
            y_pred = np.full(len(y_test), train_mean)
            m = _metrics(y_test, y_pred)
            _trained_models[name] = {"type": "baseline", "mean": train_mean}
        else:
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", estimator),
            ])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            m = _metrics(y_test, y_pred)
            _trained_models[name] = pipe

            # Store feature names for importance
            if name in ("Random Forest", "Gradient Boosting", "Decision Tree"):
                ohe = pipe.named_steps["preprocessor"].named_transformers_["cat"]
                cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
                _feature_names = NUMERIC_FEATURES + cat_names

        row = {"model": name, **m}
        results.append(row)

    _model_metrics = results

    # Cache metrics
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    return results


def get_model_metrics() -> list[dict]:
    if not _model_metrics:
        return train_all_models()
    return _model_metrics


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def predict_price(
    company: str,
    ram_gb: float,
    battery_mah: int,
    front_cam_mp: float,
    back_cam_mp: float,
    back_cam_lenses: int,
    screen_inches: float,
    weight_g: float,
    launched_year: int,
    processor_family: str,
    model_name: str = "Random Forest",
) -> dict:
    if not _trained_models:
        train_all_models()

    m = _trained_models.get(model_name)
    if m is None:
        return {"error": f"Model '{model_name}' not found. Available: {list(_trained_models.keys())}"}

    input_dict = {
        "RAM_num": [ram_gb],
        "Battery_num": [battery_mah],
        "FrontCam_num": [front_cam_mp],
        "BackCam_primary": [back_cam_mp],
        "BackCam_lenses": [back_cam_lenses],
        "Screen_num": [screen_inches],
        "Weight_num": [weight_g],
        "Launched Year": [launched_year],
        "Company Name": [company],
        "Processor_Family": [processor_family],
    }
    X_input = pd.DataFrame(input_dict)

    if isinstance(m, dict) and m.get("type") == "baseline":
        prediction = m["mean"]
        uncertainty_low = prediction * 0.8
        uncertainty_high = prediction * 1.2
    else:
        prediction = float(m.predict(X_input)[0])
        # Approximate ±1 std of residuals as uncertainty
        if _y_test is not None:
            y_pred_test = m.predict(_X_test)
            residuals_std = float(np.std(_y_test.values - y_pred_test))
            uncertainty_low = max(0, prediction - residuals_std)
            uncertainty_high = prediction + residuals_std
        else:
            uncertainty_low = prediction * 0.85
            uncertainty_high = prediction * 1.15

    # Get relevant metrics
    metrics = next((r for r in _model_metrics if r["model"] == model_name), {})

    return {
        "estimated_india_price": round(prediction, 2),
        "uncertainty_low": round(uncertainty_low, 2),
        "uncertainty_high": round(uncertainty_high, 2),
        "model_used": model_name,
        "model_metrics": metrics,
        "disclaimer": (
            "This is a machine-learning estimate based on patterns in the provided dataset "
            "and should not be treated as an actual market quotation."
        ),
    }


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------

def get_feature_importance(model_name: str = "Random Forest") -> dict:
    if not _trained_models:
        train_all_models()

    m = _trained_models.get(model_name)
    if m is None or (isinstance(m, dict) and m.get("type") == "baseline"):
        return {"error": "Feature importance not available for this model."}

    if not _feature_names:
        return {"error": "Feature names not computed yet. Retrain models."}

    estimator = m.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        pairs = sorted(
            zip(_feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )[:25]  # top 25
        return {
            "model": model_name,
            "method": "Built-in feature importance",
            "features": [{"feature": f, "importance": round(float(v), 6)} for f, v in pairs],
        }

    if hasattr(estimator, "coef_"):
        # Linear regression: use absolute coefficients on preprocessed data
        coefs = np.abs(estimator.coef_)
        pairs = sorted(zip(_feature_names, coefs), key=lambda x: x[1], reverse=True)[:25]
        return {
            "model": model_name,
            "method": "Absolute coefficient magnitude",
            "features": [{"feature": f, "importance": round(float(v), 6)} for f, v in pairs],
        }

    return {"error": "Feature importance method not available for this model."}


# ---------------------------------------------------------------------------
# Model Diagnostics
# ---------------------------------------------------------------------------

def get_diagnostics(model_name: str = "Random Forest") -> dict:
    if not _trained_models:
        train_all_models()

    m = _trained_models.get(model_name)
    if m is None:
        return {"error": f"Model '{model_name}' not found."}

    if _X_test is None or _y_test is None:
        return {"error": "Test set not available. Retrain models."}

    if isinstance(m, dict) and m.get("type") == "baseline":
        y_pred = np.full(len(_y_test), m["mean"])
    else:
        y_pred = m.predict(_X_test)

    residuals = _y_test.values - y_pred
    metrics = _metrics(_y_test, y_pred)

    # Sample up to 300 points for the chart
    n = min(300, len(_y_test))
    idx = np.random.default_rng(0).choice(len(_y_test), n, replace=False)

    return {
        "model": model_name,
        "metrics": metrics,
        "actual_vs_predicted": [
            {"actual": round(float(_y_test.iloc[i]), 2), "predicted": round(float(y_pred[i]), 2)}
            for i in idx
        ],
        "residuals": [round(float(r), 2) for r in residuals[idx]],
        "residuals_mean": round(float(np.mean(residuals)), 2),
        "residuals_std": round(float(np.std(residuals)), 2),
    }
