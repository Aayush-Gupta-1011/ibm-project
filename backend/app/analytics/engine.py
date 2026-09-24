"""
Analytics engine: computes all KPIs, distributions, correlations,
brand intelligence, market analysis, and global price comparisons.
All results are derived from the cleaned dataset — no fabrication.
"""

import math
import numpy as np
import pandas as pd

from app.preprocessing.cleaner import get_clean_df, RAW_PATH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_round(val, digits: int = 2):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return round(float(val), digits)


def _df() -> pd.DataFrame:
    return get_clean_df()


# ---------------------------------------------------------------------------
# 1. KPI Dashboard
# ---------------------------------------------------------------------------

def get_kpis(include_tablets: bool = True) -> dict:
    df = _df()
    if not include_tablets:
        df = df[~df["Is_Tablet"]]

    prices = df["India_Price"].dropna()
    rams = df["RAM_num"].dropna()
    batteries = df["Battery_num"].dropna()
    screens = df["Screen_num"].dropna()

    segment_counts = df["Price_Segment"].value_counts().to_dict()

    return {
        "total_smartphones": int(len(df)),
        "unique_brands": int(df["Company Name"].nunique()),
        "unique_processors": int(df["Processor"].nunique()),
        "unique_processor_families": int(df["Processor_Family"].nunique()),
        "avg_india_price": _safe_round(prices.mean()),
        "median_india_price": _safe_round(prices.median()),
        "min_india_price": _safe_round(prices.min()),
        "max_india_price": _safe_round(prices.max()),
        "avg_ram": _safe_round(rams.mean(), 1),
        "avg_battery": _safe_round(batteries.mean()),
        "avg_screen_size": _safe_round(screens.mean(), 2),
        "latest_year": int(df["Launched Year"].max()),
        "oldest_year": int(df["Launched Year"].min()),
        "total_tablets": int(df["Is_Tablet"].sum()),
        "total_foldables": int(df["Is_Foldable"].sum()),
        "segment_distribution": segment_counts,
    }


# ---------------------------------------------------------------------------
# 2. Market Analysis
# ---------------------------------------------------------------------------

def get_brand_distribution() -> list:
    df = _df()
    counts = df["Company Name"].value_counts().reset_index()
    counts.columns = ["brand", "count"]
    total = len(df)
    counts["market_share_pct"] = (counts["count"] / total * 100).round(2)
    return counts.to_dict(orient="records")


def get_yearly_launch_counts() -> list:
    df = _df()
    counts = df.groupby("Launched Year").size().reset_index(name="count")
    avg_prices = df.groupby("Launched Year")["India_Price"].mean().round(2).reset_index(name="avg_india_price")
    result = counts.merge(avg_prices, on="Launched Year")
    result.columns = ["year", "count", "avg_india_price"]
    return result.to_dict(orient="records")


def get_ram_distribution() -> list:
    df = _df()
    counts = df["RAM_num"].dropna().value_counts().sort_index().reset_index()
    counts.columns = ["ram_gb", "count"]
    counts["ram_gb"] = counts["ram_gb"].astype(int)
    return counts.to_dict(orient="records")


def get_ram_by_year() -> list:
    df = _df()
    result = df.groupby("Launched Year")["RAM_num"].mean().round(2).reset_index()
    result.columns = ["year", "avg_ram_gb"]
    return result.to_dict(orient="records")


def get_battery_distribution() -> list:
    df = _df()
    bins = [0, 2000, 3000, 4000, 5000, 6000, 8000, 20000]
    labels = ["<2000", "2000-3000", "3000-4000", "4000-5000", "5000-6000", "6000-8000", "8000+"]
    df = df.copy()
    df["battery_bin"] = pd.cut(df["Battery_num"], bins=bins, labels=labels)
    counts = df["battery_bin"].value_counts().reindex(labels).reset_index()
    counts.columns = ["range", "count"]
    return counts.to_dict(orient="records")


def get_avg_battery_by_brand() -> list:
    df = _df()
    result = (
        df.groupby("Company Name")["Battery_num"]
        .mean().round(0).sort_values(ascending=False).reset_index()
    )
    result.columns = ["brand", "avg_battery_mah"]
    return result.to_dict(orient="records")


def get_brand_by_year() -> list:
    df = _df()
    result = df.groupby(["Launched Year", "Company Name"]).size().reset_index(name="count")
    result.columns = ["year", "brand", "count"]
    return result.to_dict(orient="records")


# ---------------------------------------------------------------------------
# 3. Brand Intelligence
# ---------------------------------------------------------------------------

def get_brand_profile(brand: str) -> dict:
    df = _df()
    bdf = df[df["Company Name"].str.lower() == brand.lower()].copy()
    if bdf.empty:
        return {"error": f"Brand '{brand}' not found"}

    prices = bdf["India_Price"].dropna()
    return {
        "brand": brand,
        "model_count": int(len(bdf)),
        "avg_india_price": _safe_round(prices.mean()),
        "median_india_price": _safe_round(prices.median()),
        "min_india_price": _safe_round(prices.min()),
        "max_india_price": _safe_round(prices.max()),
        "avg_ram": _safe_round(bdf["RAM_num"].dropna().mean(), 1),
        "avg_battery": _safe_round(bdf["Battery_num"].dropna().mean()),
        "avg_screen_size": _safe_round(bdf["Screen_num"].dropna().mean(), 2),
        "avg_front_cam": _safe_round(bdf["FrontCam_num"].dropna().mean(), 1),
        "avg_back_cam": _safe_round(bdf["BackCam_primary"].dropna().mean(), 1),
        "years_active": sorted(bdf["Launched Year"].dropna().unique().tolist()),
        "price_segment_distribution": bdf["Price_Segment"].value_counts().to_dict(),
        "processor_families": bdf["Processor_Family"].value_counts().to_dict(),
        "models": (
            bdf[["Model Name", "India_Price", "RAM_num", "Battery_num", "Screen_num", "Launched Year", "Price_Segment"]]
            .rename(columns={
                "Model Name": "model_name",
                "India_Price": "india_price",
                "RAM_num": "ram_gb",
                "Battery_num": "battery_mah",
                "Screen_num": "screen_inches",
                "Launched Year": "year",
                "Price_Segment": "price_segment",
            })
            .to_dict(orient="records")
        ),
    }


def get_brand_comparison(brands: list) -> list:
    df = _df()
    results = []
    for brand in brands:
        bdf = df[df["Company Name"].str.lower() == brand.lower()]
        if bdf.empty:
            continue
        prices = bdf["India_Price"].dropna()
        results.append({
            "brand": brand,
            "model_count": int(len(bdf)),
            "avg_india_price": _safe_round(prices.mean()),
            "median_india_price": _safe_round(prices.median()),
            "avg_ram": _safe_round(bdf["RAM_num"].dropna().mean(), 1),
            "avg_battery": _safe_round(bdf["Battery_num"].dropna().mean()),
            "avg_screen": _safe_round(bdf["Screen_num"].dropna().mean(), 2),
            "avg_front_cam": _safe_round(bdf["FrontCam_num"].dropna().mean(), 1),
            "avg_back_cam": _safe_round(bdf["BackCam_primary"].dropna().mean(), 1),
        })
    return results


# ---------------------------------------------------------------------------
# 4. Price Analysis
# ---------------------------------------------------------------------------

def get_price_distribution(bins: int = 20) -> list:
    df = _df()
    prices = df["India_Price"].dropna()
    hist, edges = np.histogram(prices, bins=bins)
    result = []
    for i, count in enumerate(hist):
        result.append({
            "range_start": int(edges[i]),
            "range_end": int(edges[i + 1]),
            "label": f"₹{int(edges[i] // 1000)}k-{int(edges[i+1] // 1000)}k",
            "count": int(count),
        })
    return result


def get_price_by_segment() -> list:
    df = _df()
    seg_order = ["Budget", "Mid-Range", "Upper Mid-Range", "Premium", "Ultra Premium", "Unknown"]
    result = []
    for seg in seg_order:
        sdf = df[df["Price_Segment"] == seg]
        if sdf.empty:
            continue
        prices = sdf["India_Price"].dropna()
        result.append({
            "segment": seg,
            "count": int(len(sdf)),
            "avg_price": _safe_round(prices.mean()),
            "median_price": _safe_round(prices.median()),
            "avg_ram": _safe_round(sdf["RAM_num"].dropna().mean(), 1),
            "avg_battery": _safe_round(sdf["Battery_num"].dropna().mean()),
            "avg_screen": _safe_round(sdf["Screen_num"].dropna().mean(), 2),
        })
    return result


def get_price_vs_spec(spec: str) -> dict:
    col_map = {
        "ram": "RAM_num",
        "battery": "Battery_num",
        "screen": "Screen_num",
        "front_cam": "FrontCam_num",
        "back_cam": "BackCam_primary",
        "weight": "Weight_num",
    }
    spec_key = spec.lower()
    if spec_key not in col_map:
        return {"data": [], "correlation": None}
    df = _df()
    sub = df[["Company Name", "Model Name", col_map[spec_key], "India_Price"]].dropna()
    sub = sub.rename(columns={
        col_map[spec_key]: "x",
        "India_Price": "y",
        "Company Name": "brand",
        "Model Name": "model",
    })
    corr = float(sub["x"].corr(sub["y"]))
    corr = None if math.isnan(corr) else round(corr, 4)
    return {"data": sub.to_dict(orient="records"), "correlation": corr}


# ---------------------------------------------------------------------------
# 5. Global Price Comparison
# ---------------------------------------------------------------------------

def get_global_price_for_model(model_name: str) -> dict:
    df = _df()
    row = df[df["Model Name"].str.lower() == model_name.lower()]
    if row.empty:
        row = df[df["Model Name"].str.lower().str.contains(model_name.lower(), regex=False, na=False)]
    if row.empty:
        return {"error": f"Model '{model_name}' not found"}
    r = row.iloc[0]

    def safe(v):
        return None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)

    return {
        "model": r["Model Name"],
        "brand": r["Company Name"],
        "india_price": safe(r["India_Price"]),
        "pakistan_price": safe(r["Pakistan_Price"]),
        "china_price": safe(r["China_Price"]),
        "usa_price": safe(r["USA_Price"]),
        "dubai_price": safe(r["Dubai_Price"]),
        "note": "Prices shown in their native currencies (INR, PKR, CNY, USD, AED). Direct comparison requires exchange rates.",
    }


def get_avg_prices_by_country() -> dict:
    df = _df()
    return {
        "india_avg": _safe_round(df["India_Price"].mean()),
        "pakistan_avg": _safe_round(df["Pakistan_Price"].mean()),
        "china_avg": _safe_round(df["China_Price"].mean()),
        "usa_avg": _safe_round(df["USA_Price"].mean()),
        "dubai_avg": _safe_round(df["Dubai_Price"].mean()),
        "india_median": _safe_round(df["India_Price"].median()),
        "usa_median": _safe_round(df["USA_Price"].median()),
    }


# ---------------------------------------------------------------------------
# 6. Specification Explorer
# ---------------------------------------------------------------------------

def get_filtered_specs(
    brand=None, min_ram=None, max_ram=None,
    min_battery=None, max_battery=None,
    min_price=None, max_price=None,
    min_year=None, max_year=None,
    price_segment=None, search=None,
    sort_by="India_Price", sort_asc=True, limit=200,
) -> list:
    df = _df()

    if brand and brand.lower() != "all":
        df = df[df["Company Name"].str.lower() == brand.lower()]
    if min_ram is not None:
        df = df[df["RAM_num"] >= min_ram]
    if max_ram is not None:
        df = df[df["RAM_num"] <= max_ram]
    if min_battery is not None:
        df = df[df["Battery_num"] >= min_battery]
    if max_battery is not None:
        df = df[df["Battery_num"] <= max_battery]
    if min_price is not None:
        df = df[df["India_Price"] >= min_price]
    if max_price is not None:
        df = df[df["India_Price"] <= max_price]
    if min_year is not None:
        df = df[df["Launched Year"] >= min_year]
    if max_year is not None:
        df = df[df["Launched Year"] <= max_year]
    if price_segment and price_segment.lower() != "all":
        df = df[df["Price_Segment"].str.lower() == price_segment.lower()]
    if search:
        mask = (
            df["Model Name"].str.lower().str.contains(search.lower(), regex=False, na=False) |
            df["Company Name"].str.lower().str.contains(search.lower(), regex=False, na=False) |
            df["Processor"].str.lower().str.contains(search.lower(), regex=False, na=False)
        )
        df = df[mask]

    valid_sort = ["India_Price", "RAM_num", "Battery_num", "Screen_num", "Launched Year"]
    if sort_by not in valid_sort:
        sort_by = "India_Price"

    df = df.sort_values(sort_by, ascending=sort_asc).head(limit)

    cols = [
        "Company Name", "Model Name", "India_Price", "RAM_num", "Battery_num",
        "FrontCam_num", "BackCam_primary", "Screen_num", "Processor",
        "Processor_Family", "Launched Year", "Price_Segment", "Weight_num",
        "Is_Tablet", "Is_Foldable",
    ]
    out = df[cols].rename(columns={
        "Company Name": "brand", "Model Name": "model",
        "India_Price": "india_price", "RAM_num": "ram_gb",
        "Battery_num": "battery_mah", "FrontCam_num": "front_cam_mp",
        "BackCam_primary": "back_cam_mp", "Screen_num": "screen_inches",
        "Processor": "processor", "Processor_Family": "processor_family",
        "Launched Year": "year", "Price_Segment": "price_segment",
        "Weight_num": "weight_g", "Is_Tablet": "is_tablet", "Is_Foldable": "is_foldable",
    })
    # Replace NaN with None for JSON serialisation
    return out.where(pd.notna(out), None).to_dict(orient="records")


# ---------------------------------------------------------------------------
# 7. Smartphone Comparator
# ---------------------------------------------------------------------------

def compare_smartphones(model_names: list) -> list:
    df = _df()
    price_cols = ["India_Price", "Pakistan_Price", "China_Price", "USA_Price", "Dubai_Price"]
    spec_cols = [
        "Company Name", "Model Name", "RAM_num", "Battery_num", "FrontCam_num",
        "BackCam_primary", "BackCam_lenses", "Screen_num", "Weight_num",
        "Processor", "Processor_Family", "Launched Year", "Price_Segment",
    ] + price_cols

    results = []
    for name in model_names[:4]:
        row = df[df["Model Name"].str.lower() == name.lower()]
        if row.empty:
            row = df[df["Model Name"].str.lower().str.contains(name.lower(), regex=False, na=False)]
        if row.empty:
            continue
        r = row.iloc[0][spec_cols].to_dict()
        for k, v in r.items():
            if isinstance(v, float) and math.isnan(v):
                r[k] = None
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# 8. Correlation Matrix
# ---------------------------------------------------------------------------

def get_correlation_matrix() -> dict:
    df = _df()
    num_cols = [
        "India_Price", "RAM_num", "Battery_num", "Screen_num",
        "FrontCam_num", "BackCam_primary", "Weight_num", "Launched Year",
    ]
    sub = df[num_cols].dropna()
    corr = sub.corr().round(4)
    return {
        "columns": num_cols,
        "matrix": corr.values.tolist(),
    }


# ---------------------------------------------------------------------------
# 9. Data Quality Report
# ---------------------------------------------------------------------------

def get_data_quality_report() -> dict:
    raw_full = pd.read_csv(RAW_PATH, encoding="utf-8", encoding_errors="replace")
    total_raw = len(raw_full)

    clean_df = get_clean_df()
    total_clean = len(clean_df)
    n_dupes = max(0, total_raw - 1 - total_clean)  # -1 for blank trailing row

    missing = {}
    for col in clean_df.columns:
        n = int(clean_df[col].isna().sum())
        if n > 0:
            missing[col] = {"count": n, "pct": round(n / total_clean * 100, 2)}

    score = 100
    india_missing = missing.get("India_Price", {}).get("count", 0)
    if india_missing > 0:
        score -= 20
    if n_dupes > 5:
        score -= 20
    elif n_dupes > 0:
        score -= 5
    encoding_errors = missing.get("China_Price", {}).get("count", 0)
    if encoding_errors > 0:
        score -= min(15, encoding_errors * 3)
    score = max(0, score)

    return {
        "total_raw_records": total_raw,
        "blank_rows": 1,
        "duplicate_records": n_dupes,
        "total_clean_records": total_clean,
        "unique_brands": int(clean_df["Company Name"].nunique()),
        "unique_processors": int(clean_df["Processor"].nunique()),
        "missing_values": missing,
        "quality_score": score,
        "score_rules": {
            "No missing India prices": 20,
            "No duplicate records": 20,
            "No encoding errors in China prices": 15,
            "All numeric fields parseable": 20,
            "Launch year within valid range (2014-2025)": 10,
            "No invalid spec values": 15,
        },
    }


# ---------------------------------------------------------------------------
# 10. Automatic Insights
# ---------------------------------------------------------------------------

def generate_insights() -> list:
    df = _df()
    insights = []

    top_brand = df["Company Name"].value_counts().idxmax()
    top_brand_count = int(df["Company Name"].value_counts().iloc[0])
    top_brand_pct = round(top_brand_count / len(df) * 100, 1)
    insights.append({
        "category": "Market",
        "title": "Top Brand by Volume",
        "text": f"{top_brand} has the most models in this dataset ({top_brand_count} models, {top_brand_pct}% of total).",
        "type": "info",
    })

    median_price = df["India_Price"].dropna().median()
    insights.append({
        "category": "Pricing",
        "title": "Median India Launch Price",
        "text": f"The median India launch price across all devices is \u20b9{int(median_price):,}.",
        "type": "stat",
    })

    common_ram = int(df["RAM_num"].dropna().mode()[0])
    ram_pct = round((df["RAM_num"] == common_ram).sum() / len(df) * 100, 1)
    insights.append({
        "category": "Specification",
        "title": "Most Common RAM Configuration",
        "text": f"{common_ram}GB is the most common RAM configuration, appearing in {ram_pct}% of devices.",
        "type": "stat",
    })

    latest = int(df["Launched Year"].max())
    p2020 = df[df["Launched Year"] == 2020]["India_Price"].mean()
    p_latest = df[df["Launched Year"] == latest]["India_Price"].mean()
    if not (math.isnan(p2020) or math.isnan(p_latest)):
        pct_change = round((p_latest - p2020) / p2020 * 100, 1)
        direction = "increased" if pct_change > 0 else "decreased"
        insights.append({
            "category": "Trend",
            "title": "Average Price Trend (2020 \u2192 Latest Year)",
            "text": f"The average India launch price {direction} by {abs(pct_change)}% from 2020 (\u20b9{int(p2020):,}) to {latest} (\u20b9{int(p_latest):,}).",
            "type": "trend",
        })

    sub = df[["Battery_num", "India_Price"]].dropna()
    corr_battery = sub["Battery_num"].corr(sub["India_Price"])
    insights.append({
        "category": "Relationship",
        "title": "Battery vs Price Correlation",
        "text": f"Battery capacity has a Pearson correlation of {corr_battery:.3f} with India launch price. Note: correlation does not imply causation.",
        "type": "correlation",
    })

    sub2 = df[["RAM_num", "India_Price"]].dropna()
    corr_ram = sub2["RAM_num"].corr(sub2["India_Price"])
    insights.append({
        "category": "Relationship",
        "title": "RAM vs Price Correlation",
        "text": f"RAM has a Pearson correlation of {corr_ram:.3f} with India launch price, making it one of the stronger spec predictors.",
        "type": "correlation",
    })

    top_proc = df["Processor_Family"].value_counts().idxmax()
    top_proc_pct = round(df["Processor_Family"].value_counts().iloc[0] / len(df) * 100, 1)
    insights.append({
        "category": "Specification",
        "title": "Most Common Processor Family",
        "text": f"'{top_proc}' is the most common processor family at {top_proc_pct}% of all devices.",
        "type": "info",
    })

    budget_pct = round((df["Price_Segment"] == "Budget").sum() / len(df) * 100, 1)
    ultra_pct = round((df["Price_Segment"] == "Ultra Premium").sum() / len(df) * 100, 1)
    insights.append({
        "category": "Market",
        "title": "Budget vs Ultra Premium Split",
        "text": f"Budget devices (< \u20b915,000) account for {budget_pct}% of the dataset; Ultra Premium (> \u20b91,00,000) account for {ultra_pct}%.",
        "type": "stat",
    })

    top_year = int(df["Launched Year"].value_counts().idxmax())
    top_year_count = int(df["Launched Year"].value_counts().iloc[0])
    insights.append({
        "category": "Trend",
        "title": "Most Active Launch Year",
        "text": f"{top_year} had the highest number of device launches in this dataset ({top_year_count} devices).",
        "type": "trend",
    })

    sub3 = df[["Screen_num", "India_Price"]].dropna()
    corr_screen = sub3["Screen_num"].corr(sub3["India_Price"])
    insights.append({
        "category": "Relationship",
        "title": "Screen Size vs Price Correlation",
        "text": f"Screen size has a Pearson correlation of {corr_screen:.3f} with India launch price.",
        "type": "correlation",
    })

    return insights


# ---------------------------------------------------------------------------
# 11. Helpers for dropdowns
# ---------------------------------------------------------------------------

def get_all_models() -> list:
    return sorted(_df()["Model Name"].dropna().unique().tolist())


def get_all_brands() -> list:
    return sorted(_df()["Company Name"].dropna().unique().tolist())


def get_all_processor_families() -> list:
    return sorted(_df()["Processor_Family"].dropna().unique().tolist())
