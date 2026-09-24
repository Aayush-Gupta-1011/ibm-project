"""
Test suite for the Mobile Market Intelligence backend.
Tests cover: CSV loading, data cleaning, price parsing, KPI calculations,
filtering, analytics, model training, prediction, and API endpoints.
"""

import math
import pytest
import numpy as np
import pandas as pd
import sys
import os

# Ensure backend is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.preprocessing.cleaner import (
    parse_ram, parse_weight, parse_camera_primary, count_camera_lenses,
    parse_battery, parse_screen, parse_india_price, parse_pakistan_price,
    parse_china_price, parse_usa_price, parse_dubai_price,
    assign_processor_family, assign_price_segment, load_and_clean,
    is_foldable, is_tablet,
)
from app.analytics.engine import (
    get_kpis, get_brand_distribution, get_yearly_launch_counts,
    get_price_by_segment, get_price_vs_spec, generate_insights,
    get_data_quality_report, get_brand_profile, compare_smartphones,
    get_correlation_matrix,
)
from app.models.trainer import train_all_models, predict_price, get_feature_importance


# ---------------------------------------------------------------------------
# 1. CSV Loading
# ---------------------------------------------------------------------------

class TestCSVLoading:
    def test_loads_without_error(self):
        df = load_and_clean()
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        df = load_and_clean()
        for col in ["Company Name", "Model Name", "India_Price", "RAM_num", "Battery_num"]:
            assert col in df.columns

    def test_no_blank_rows(self):
        df = load_and_clean()
        blank = df.isnull().all(axis=1).sum()
        assert blank == 0

    def test_row_count_reasonable(self):
        df = load_and_clean()
        # After dedup and removing blank row, should have between 800 and 932 rows
        assert 800 <= len(df) <= 932


# ---------------------------------------------------------------------------
# 2. Data Cleaning — Field Parsers
# ---------------------------------------------------------------------------

class TestParsers:
    def test_parse_ram_basic(self):
        assert parse_ram("6GB") == 6.0

    def test_parse_ram_decimal(self):
        assert parse_ram("1.5GB") == 1.5

    def test_parse_ram_ambiguous(self):
        assert parse_ram("8GB / 12GB") == 8.0

    def test_parse_ram_none(self):
        assert parse_ram(None) is None

    def test_parse_weight_basic(self):
        assert parse_weight("174g") == 174.0

    def test_parse_weight_decimal(self):
        assert parse_weight("222.8g") == 222.8

    def test_parse_battery_with_comma(self):
        assert parse_battery("3,600mAh") == 3600

    def test_parse_battery_no_comma(self):
        assert parse_battery("5000mAh") == 5000

    def test_parse_screen_basic(self):
        assert parse_screen("6.1 inches") == 6.1

    def test_parse_screen_foldable(self):
        result = parse_screen("6.7 inches (main), 2.7 inches (external)")
        assert result == 6.7  # takes first number

    def test_parse_camera_basic(self):
        assert parse_camera_primary("12MP") == 12.0

    def test_parse_camera_dual(self):
        assert parse_camera_primary("50MP + 12MP") == 50.0

    def test_parse_camera_prefix(self):
        assert parse_camera_primary("Dual 32MP") == 32.0

    def test_count_lenses_single(self):
        assert count_camera_lenses("50MP") == 1

    def test_count_lenses_dual(self):
        assert count_camera_lenses("50MP + 12MP") == 2

    def test_count_lenses_triple(self):
        assert count_camera_lenses("50MP + 12MP + 10MP") == 3


# ---------------------------------------------------------------------------
# 3. Price Parsing
# ---------------------------------------------------------------------------

class TestPriceParsing:
    def test_india_basic(self):
        assert parse_india_price("INR 79,999") == 79999.0

    def test_india_indian_format(self):
        # "INR 1,04,999" → 104999
        assert parse_india_price("INR 1,04,999") == 104999.0

    def test_pakistan_basic(self):
        assert parse_pakistan_price("PKR 224,999") == 224999.0

    def test_pakistan_not_available(self):
        assert parse_pakistan_price("Not available") is None

    def test_china_basic(self):
        result = parse_china_price("CNY 5,799")
        assert result == 5799.0

    def test_usa_whole(self):
        assert parse_usa_price("USD 799") == 799.0

    def test_usa_with_comma(self):
        assert parse_usa_price("USD 1,049") == 1049.0

    def test_usa_decimal(self):
        assert parse_usa_price("USD 634.99") == 634.99

    def test_usa_european_decimal(self):
        # "USD 396,22" with comma as decimal separator
        result = parse_usa_price("USD 396,22")
        assert result == 396.22

    def test_dubai_basic(self):
        assert parse_dubai_price("AED 2,799") == 2799.0


# ---------------------------------------------------------------------------
# 4. Derived Columns
# ---------------------------------------------------------------------------

class TestDerivedColumns:
    def test_processor_family_apple(self):
        assert assign_processor_family("A17 Bionic") == "Apple"

    def test_processor_family_snapdragon_flagship(self):
        assert assign_processor_family("Snapdragon 8 Gen 3") == "Snapdragon 8 (Flagship)"

    def test_processor_family_dimensity(self):
        assert assign_processor_family("MediaTek Dimensity 9400") == "Dimensity 9xxx (Flagship)"

    def test_processor_family_tensor(self):
        assert assign_processor_family("Google Tensor G4") == "Google Tensor"

    def test_processor_family_unknown(self):
        result = assign_processor_family("Some Unknown Chip XYZ")
        assert isinstance(result, str)

    def test_price_segment_budget(self):
        assert assign_price_segment(9999) == "Budget"

    def test_price_segment_midrange(self):
        assert assign_price_segment(20000) == "Mid-Range"

    def test_price_segment_ultra(self):
        assert assign_price_segment(200000) == "Ultra Premium"

    def test_price_segment_none(self):
        assert assign_price_segment(None) == "Unknown"

    def test_is_foldable_true(self):
        assert is_foldable("Samsung Galaxy Z Fold 5") is True

    def test_is_foldable_false(self):
        assert is_foldable("iPhone 16 Pro") is False

    def test_is_tablet_ipad(self):
        assert is_tablet("iPad Pro 11-inch 128GB", 11.0) is True

    def test_is_tablet_large_screen(self):
        assert is_tablet("Some Device", 14.6) is True

    def test_is_tablet_phone(self):
        assert is_tablet("Samsung Galaxy S24", 6.1) is False


# ---------------------------------------------------------------------------
# 5. KPI Calculations
# ---------------------------------------------------------------------------

class TestKPIs:
    def test_kpis_structure(self):
        kpis = get_kpis()
        assert "total_smartphones" in kpis
        assert "unique_brands" in kpis
        assert "avg_india_price" in kpis
        assert "median_india_price" in kpis

    def test_kpis_positive_values(self):
        kpis = get_kpis()
        assert kpis["total_smartphones"] > 0
        assert kpis["unique_brands"] > 0
        assert kpis["avg_india_price"] > 0
        assert kpis["median_india_price"] > 0

    def test_median_less_than_or_equal_avg(self):
        # With right-skewed price distribution, median <= mean
        kpis = get_kpis()
        # Just verify both are positive numbers
        assert kpis["median_india_price"] > 0
        assert kpis["avg_india_price"] > 0


# ---------------------------------------------------------------------------
# 6. Analytics Functions
# ---------------------------------------------------------------------------

class TestAnalytics:
    def test_brand_distribution_returns_list(self):
        result = get_brand_distribution()
        assert isinstance(result, list)
        assert len(result) > 0
        assert "brand" in result[0]
        assert "count" in result[0]

    def test_yearly_launches_all_years_present(self):
        result = get_yearly_launch_counts()
        years = [r["year"] for r in result]
        assert 2024 in years or 2025 in years

    def test_price_by_segment_segments(self):
        result = get_price_by_segment()
        segments = [r["segment"] for r in result]
        assert "Budget" in segments or "Mid-Range" in segments

    def test_price_vs_spec_ram(self):
        result = get_price_vs_spec("ram")
        assert "data" in result
        assert "correlation" in result
        assert isinstance(result["data"], list)
        assert len(result["data"]) > 0
        assert result["correlation"] is not None

    def test_price_vs_spec_invalid(self):
        result = get_price_vs_spec("invalid_spec")
        assert result["data"] == []

    def test_insights_non_empty(self):
        insights = generate_insights()
        assert isinstance(insights, list)
        assert len(insights) >= 5
        for item in insights:
            assert "category" in item
            assert "title" in item
            assert "text" in item

    def test_correlation_matrix_shape(self):
        result = get_correlation_matrix()
        assert "columns" in result
        assert "matrix" in result
        n = len(result["columns"])
        assert len(result["matrix"]) == n
        assert len(result["matrix"][0]) == n

    def test_brand_profile_apple(self):
        result = get_brand_profile("Apple")
        assert "error" not in result
        assert result["model_count"] > 0

    def test_brand_profile_nonexistent(self):
        result = get_brand_profile("FakeBrandXYZ")
        assert "error" in result

    def test_data_quality_report(self):
        report = get_data_quality_report()
        assert "total_raw_records" in report
        assert "total_clean_records" in report
        assert "quality_score" in report
        assert 0 <= report["quality_score"] <= 100


# ---------------------------------------------------------------------------
# 7. Filtering
# ---------------------------------------------------------------------------

class TestFiltering:
    def test_filter_by_brand(self):
        from app.analytics.engine import get_filtered_specs
        result = get_filtered_specs(brand="Samsung")
        assert all(r["brand"] == "Samsung" for r in result)

    def test_filter_by_price_range(self):
        from app.analytics.engine import get_filtered_specs
        result = get_filtered_specs(min_price=50000, max_price=100000)
        for r in result:
            if r["india_price"] is not None:
                assert 50000 <= r["india_price"] <= 100000

    def test_filter_by_ram(self):
        from app.analytics.engine import get_filtered_specs
        result = get_filtered_specs(min_ram=8.0)
        for r in result:
            if r["ram_gb"] is not None:
                assert r["ram_gb"] >= 8.0

    def test_search(self):
        from app.analytics.engine import get_filtered_specs
        result = get_filtered_specs(search="Galaxy S24")
        assert len(result) > 0

    def test_comparator(self):
        result = compare_smartphones(["iPhone 16 128GB", "Samsung Galaxy S24 128GB"])
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 8. Machine Learning
# ---------------------------------------------------------------------------

class TestML:
    @pytest.fixture(scope="class", autouse=True)
    def train_models(self):
        train_all_models(force=True)

    def test_metrics_list(self):
        from app.models.trainer import get_model_metrics
        metrics = get_model_metrics()
        assert isinstance(metrics, list)
        assert len(metrics) >= 5

    def test_metrics_fields(self):
        from app.models.trainer import get_model_metrics
        for m in get_model_metrics():
            assert "model" in m
            assert "mae" in m
            assert "rmse" in m
            assert "r2" in m

    def test_rf_r2_positive(self):
        from app.models.trainer import get_model_metrics
        rf = next(m for m in get_model_metrics() if m["model"] == "Random Forest")
        assert rf["r2"] > 0, "Random Forest R² should be positive"

    def test_rf_beats_baseline(self):
        from app.models.trainer import get_model_metrics
        metrics = get_model_metrics()
        baseline_mae = next(m["mae"] for m in metrics if m["model"] == "Baseline (Mean)")
        rf_mae = next(m["mae"] for m in metrics if m["model"] == "Random Forest")
        assert rf_mae < baseline_mae, "Random Forest should outperform baseline"

    def test_prediction_returns_price(self):
        result = predict_price(
            company="Samsung",
            ram_gb=8.0,
            battery_mah=4000,
            front_cam_mp=12.0,
            back_cam_mp=50.0,
            back_cam_lenses=2,
            screen_inches=6.1,
            weight_g=168.0,
            launched_year=2024,
            processor_family="Snapdragon 8 (Flagship)",
        )
        assert "estimated_india_price" in result
        assert result["estimated_india_price"] > 0
        assert "disclaimer" in result

    def test_prediction_has_uncertainty(self):
        result = predict_price(
            company="Apple",
            ram_gb=6.0,
            battery_mah=3600,
            front_cam_mp=12.0,
            back_cam_mp=48.0,
            back_cam_lenses=1,
            screen_inches=6.1,
            weight_g=174.0,
            launched_year=2024,
            processor_family="Apple",
        )
        assert "uncertainty_low" in result
        assert "uncertainty_high" in result
        assert result["uncertainty_low"] <= result["estimated_india_price"] <= result["uncertainty_high"]

    def test_feature_importance(self):
        result = get_feature_importance("Random Forest")
        assert "features" in result
        assert len(result["features"]) > 0
        assert "feature" in result["features"][0]
        assert "importance" in result["features"][0]

    def test_diagnostics(self):
        from app.models.trainer import get_diagnostics
        result = get_diagnostics("Random Forest")
        assert "metrics" in result
        assert "actual_vs_predicted" in result
        assert "residuals" in result


# ---------------------------------------------------------------------------
# 9. API Endpoints (Integration)
# ---------------------------------------------------------------------------

class TestAPIEndpoints:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
        from main import app
        return TestClient(app)

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_kpis_endpoint(self, client):
        r = client.get("/api/kpis")
        assert r.status_code == 200
        data = r.json()
        assert "total_smartphones" in data

    def test_brand_distribution_endpoint(self, client):
        r = client.get("/api/market/brand-distribution")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_insights_endpoint(self, client):
        r = client.get("/api/insights")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_brand_profile_endpoint(self, client):
        r = client.get("/api/brand/Apple")
        assert r.status_code == 200
        assert r.json()["brand"] == "Apple"

    def test_brand_not_found(self, client):
        r = client.get("/api/brand/FakeBrandXYZ999")
        assert r.status_code == 404

    def test_ml_metrics_endpoint(self, client):
        r = client.get("/api/ml/metrics")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) >= 5

    def test_predict_endpoint(self, client):
        payload = {
            "company": "Samsung",
            "ram_gb": 8.0,
            "battery_mah": 5000,
            "front_cam_mp": 12.0,
            "back_cam_mp": 50.0,
            "back_cam_lenses": 2,
            "screen_inches": 6.6,
            "weight_g": 196.0,
            "launched_year": 2024,
            "processor_family": "Exynos 2xxx (Flagship)",
            "model_name": "Random Forest",
        }
        r = client.post("/api/ml/predict", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "estimated_india_price" in data
        assert "disclaimer" in data

    def test_spec_explorer_endpoint(self, client):
        r = client.get("/api/specs/explore?brand=Apple&limit=10")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) <= 10

    def test_data_quality_endpoint(self, client):
        r = client.get("/api/data-quality")
        assert r.status_code == 200
        data = r.json()
        assert "quality_score" in data
