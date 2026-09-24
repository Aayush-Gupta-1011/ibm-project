"""
FastAPI main application — Mobile Market Intelligence & Price Prediction Platform.
All endpoints serve data derived from the actual cleaned dataset.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from app.analytics import engine
from app.models import trainer


# ---------------------------------------------------------------------------
# Startup: pre-warm data and models
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load cleaned dataset and train models on startup
    try:
        engine._df()  # warm up cache
        trainer.train_all_models()
    except Exception as e:
        print(f"[WARNING] Startup warm-up error: {e}")
    yield


app = FastAPI(
    title="Mobile Market Intelligence API",
    description="Analytics and ML prediction API for Mobiles Dataset (2025)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "Mobile Market Intelligence API"}


# ---------------------------------------------------------------------------
# Overview / KPIs
# ---------------------------------------------------------------------------

@app.get("/api/kpis")
def get_kpis(include_tablets: bool = True):
    return engine.get_kpis(include_tablets=include_tablets)


# ---------------------------------------------------------------------------
# Market Analysis
# ---------------------------------------------------------------------------

@app.get("/api/market/brand-distribution")
def brand_distribution():
    return engine.get_brand_distribution()


@app.get("/api/market/yearly-launches")
def yearly_launches():
    return engine.get_yearly_launch_counts()


@app.get("/api/market/ram-distribution")
def ram_distribution():
    return engine.get_ram_distribution()


@app.get("/api/market/ram-by-year")
def ram_by_year():
    return engine.get_ram_by_year()


@app.get("/api/market/battery-distribution")
def battery_distribution():
    return engine.get_battery_distribution()


@app.get("/api/market/battery-by-brand")
def battery_by_brand():
    return engine.get_avg_battery_by_brand()


@app.get("/api/market/brand-by-year")
def brand_by_year():
    return engine.get_brand_by_year()


# ---------------------------------------------------------------------------
# Brand Intelligence
# ---------------------------------------------------------------------------

@app.get("/api/brand/{brand_name}")
def brand_profile(brand_name: str):
    result = engine.get_brand_profile(brand_name)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/api/brands/compare")
def brand_compare(brands: str = Query(..., description="Comma-separated brand names")):
    brand_list = [b.strip() for b in brands.split(",")]
    return engine.get_brand_comparison(brand_list)


@app.get("/api/brands")
def all_brands():
    return engine.get_all_brands()


# ---------------------------------------------------------------------------
# Price Analysis
# ---------------------------------------------------------------------------

@app.get("/api/price/distribution")
def price_distribution(bins: int = 20):
    return engine.get_price_distribution(bins=bins)


@app.get("/api/price/by-segment")
def price_by_segment():
    return engine.get_price_by_segment()


@app.get("/api/price/vs-spec/{spec}")
def price_vs_spec(spec: str):
    """spec: ram | battery | screen | front_cam | back_cam | weight"""
    result = engine.get_price_vs_spec(spec)
    return result


@app.get("/api/price/country-averages")
def country_averages():
    return engine.get_avg_prices_by_country()


# ---------------------------------------------------------------------------
# Global Price Comparison
# ---------------------------------------------------------------------------

@app.get("/api/global/model")
def global_price_model(model: str = Query(..., description="Model name")):
    result = engine.get_global_price_for_model(model)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/api/global/models")
def all_models():
    return engine.get_all_models()


# ---------------------------------------------------------------------------
# Specification Explorer
# ---------------------------------------------------------------------------

@app.get("/api/specs/explore")
def explore_specs(
    brand: Optional[str] = None,
    min_ram: Optional[float] = None,
    max_ram: Optional[float] = None,
    min_battery: Optional[int] = None,
    max_battery: Optional[int] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    price_segment: Optional[str] = None,
    search: Optional[str] = None,
    sort_by: str = "India_Price",
    sort_asc: bool = True,
    limit: int = 200,
):
    return engine.get_filtered_specs(
        brand=brand, min_ram=min_ram, max_ram=max_ram,
        min_battery=min_battery, max_battery=max_battery,
        min_price=min_price, max_price=max_price,
        min_year=min_year, max_year=max_year,
        price_segment=price_segment, search=search,
        sort_by=sort_by, sort_asc=sort_asc, limit=limit,
    )


# ---------------------------------------------------------------------------
# Smartphone Comparator
# ---------------------------------------------------------------------------

@app.get("/api/compare")
def compare_phones(models: str = Query(..., description="Comma-separated model names (max 4)")):
    model_list = [m.strip() for m in models.split(",")]
    return engine.compare_smartphones(model_list)


# ---------------------------------------------------------------------------
# Correlation Matrix
# ---------------------------------------------------------------------------

@app.get("/api/analytics/correlation-matrix")
def correlation_matrix():
    return engine.get_correlation_matrix()


# ---------------------------------------------------------------------------
# Data Quality
# ---------------------------------------------------------------------------

@app.get("/api/data-quality")
def data_quality():
    return engine.get_data_quality_report()


# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------

@app.get("/api/insights")
def insights():
    return engine.generate_insights()


# ---------------------------------------------------------------------------
# ML — Model Metrics
# ---------------------------------------------------------------------------

@app.get("/api/ml/metrics")
def ml_metrics():
    return trainer.get_model_metrics()


# ---------------------------------------------------------------------------
# ML — Prediction
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    company: str
    ram_gb: float
    battery_mah: int
    front_cam_mp: float
    back_cam_mp: float
    back_cam_lenses: int = 1
    screen_inches: float
    weight_g: float
    launched_year: int
    processor_family: str
    model_name: str = "Random Forest"


@app.post("/api/ml/predict")
def predict_price(req: PredictionRequest):
    result = trainer.predict_price(
        company=req.company,
        ram_gb=req.ram_gb,
        battery_mah=req.battery_mah,
        front_cam_mp=req.front_cam_mp,
        back_cam_mp=req.back_cam_mp,
        back_cam_lenses=req.back_cam_lenses,
        screen_inches=req.screen_inches,
        weight_g=req.weight_g,
        launched_year=req.launched_year,
        processor_family=req.processor_family,
        model_name=req.model_name,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ---------------------------------------------------------------------------
# ML — Feature Importance
# ---------------------------------------------------------------------------

@app.get("/api/ml/feature-importance")
def feature_importance(model_name: str = "Random Forest"):
    result = trainer.get_feature_importance(model_name)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ---------------------------------------------------------------------------
# ML — Diagnostics
# ---------------------------------------------------------------------------

@app.get("/api/ml/diagnostics")
def diagnostics(model_name: str = "Random Forest"):
    result = trainer.get_diagnostics(model_name)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ---------------------------------------------------------------------------
# Processor Families (for prediction form)
# ---------------------------------------------------------------------------

@app.get("/api/processor-families")
def processor_families():
    return engine.get_all_processor_families()
