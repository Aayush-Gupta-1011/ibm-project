"""
Data cleaning and preprocessing pipeline for Mobiles Dataset (2025).
All transformations are documented and reproducible.
Raw columns are preserved; new analytical columns are added.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

RAW_PATH = Path(__file__).parents[3] / "data" / "raw" / "Mobiles Dataset (2025).csv"
PROCESSED_PATH = Path(__file__).parents[3] / "data" / "processed" / "mobiles_clean.csv"


# ---------------------------------------------------------------------------
# Individual field parsers
# ---------------------------------------------------------------------------

def parse_ram(val: str) -> float | None:
    """'6GB' -> 6.0,  '1.5GB' -> 1.5,  '8GB / 12GB' -> 8.0 (take lower)"""
    if pd.isna(val):
        return None
    val = str(val).strip()
    numbers = re.findall(r"\d+\.?\d*", val)
    if numbers:
        return float(numbers[0])
    return None


def parse_weight(val: str) -> float | None:
    """'174g' -> 174.0,  '300.5g' -> 300.5"""
    if pd.isna(val):
        return None
    numbers = re.findall(r"\d+\.?\d*", str(val))
    if numbers:
        return float(numbers[0])
    return None


def parse_camera_primary(val: str) -> float | None:
    """
    Extract primary (first) MP figure.
    '50MP + 12MP' -> 50.0
    '12MP / 4K'   -> 12.0
    'Dual 32MP'   -> 32.0
    '12.2MP'      -> 12.2
    '60MP + 8MP'  -> 60.0   (Huawei front dual)
    'Dual 60MP'   -> 60.0
    """
    if pd.isna(val):
        return None
    val = str(val).strip()
    # Remove qualifiers like "Dual ", "dual "
    val = re.sub(r"(?i)dual\s*", "", val)
    numbers = re.findall(r"\d+\.?\d*", val)
    if numbers:
        return float(numbers[0])
    return None


def count_camera_lenses(val: str) -> int:
    """Count number of cameras separated by ' + '"""
    if pd.isna(val):
        return 1
    parts = str(val).split("+")
    return len(parts)


def parse_battery(val: str) -> int | None:
    """'3,600mAh' -> 3600,  '5000mAh' -> 5000"""
    if pd.isna(val):
        return None
    cleaned = re.sub(r"[^\d]", "", str(val))
    if cleaned:
        return int(cleaned)
    return None


def parse_screen(val: str) -> float | None:
    """
    '6.1 inches' -> 6.1
    '6.7 inches (main), 2.7 inches (external)' -> 6.7
    '8.0 inches (unfolded)' -> 8.0
    """
    if pd.isna(val):
        return None
    numbers = re.findall(r"\d+\.?\d*", str(val))
    if numbers:
        return float(numbers[0])
    return None


def is_foldable(model_name: str) -> bool:
    """Detect folding form-factor devices."""
    keywords = ["fold", "flip", "xs 2", "open", "magic v", "magic vs", "find n", "mate x", "razr", "pocket"]
    name_lower = str(model_name).lower()
    return any(kw in name_lower for kw in keywords)


def is_tablet(model_name: str, screen_size: float | None) -> bool:
    """Detect tablets/pads based on model name or screen size > 9 inches."""
    keywords = ["ipad", "tab ", "galaxy tab", "pad ", "megapad", "xpad", "matePad", "matepad"]
    name_lower = str(model_name).lower()
    if any(kw in name_lower for kw in keywords):
        return True
    if screen_size is not None and screen_size > 9.0:
        return True
    return False


def parse_india_price(val: str) -> float | None:
    """
    'INR 79,999'     -> 79999
    'INR 1,04,999'   -> 104999  (Indian number format with 2-digit grouping)
    """
    if pd.isna(val):
        return None
    cleaned = re.sub(r"[^\d]", "", str(val))
    if cleaned:
        return float(cleaned)
    return None


def parse_pakistan_price(val: str) -> float | None:
    """'PKR 224,999' -> 224999, 'Not available' -> None"""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s.lower() in ("not available", "n/a", ""):
        return None
    cleaned = re.sub(r"[^\d]", "", s)
    if cleaned:
        return float(cleaned)
    return None


def parse_china_price(val: str) -> float | None:
    """'CNY 5,799' -> 5799, handles encoding errors"""
    if pd.isna(val):
        return None
    s = str(val).strip()
    # Strip any non-ASCII prefix/suffix characters (encoding artifacts)
    cleaned = re.sub(r"[^\d,.]", "", s)
    cleaned = cleaned.replace(",", "").replace(".", "")
    if cleaned:
        return float(cleaned)
    return None


def parse_usa_price(val: str) -> float | None:
    """'USD 799' -> 799.0, 'USD 1,049' -> 1049.0, 'USD 634.99' -> 634.99, 'USD 396,22' -> 396.22"""
    if pd.isna(val):
        return None
    s = str(val).strip()
    # Remove 'USD' and whitespace
    s = re.sub(r"(?i)usd\s*", "", s).strip()
    # Handle European decimal (comma as decimal separator) e.g. "396,22"
    # If it matches pattern digits,2digits at end → treat comma as decimal
    if re.match(r"^\d+,\d{2}$", s):
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def parse_dubai_price(val: str) -> float | None:
    """'AED 2,799' -> 2799.0"""
    if pd.isna(val):
        return None
    cleaned = re.sub(r"[^\d.]", "", str(val))
    if cleaned:
        return float(cleaned)
    return None


def assign_processor_family(processor: str) -> str:
    """Group 200+ processor strings into ~15 families for ML encoding."""
    if pd.isna(processor):
        return "Unknown"
    p = str(processor).lower()

    if "a1" in p and "bionic" in p:  # A11, A12, A13, A14, A15, A16, A17
        return "Apple"
    if "a12z" in p or "apple" in p:
        return "Apple"
    if "snapdragon 8 elite" in p or "snapdragon 8 gen 3" in p or "snapdragon 8 gen 2" in p:
        return "Snapdragon 8 (Flagship)"
    if "snapdragon 8+ gen" in p or "snapdragon 888" in p or "snapdragon 865" in p or "snapdragon 870" in p or "snapdragon 860" in p or "snapdragon 855" in p or "snapdragon 845" in p or "snapdragon 835" in p:
        return "Snapdragon 8 (Flagship)"
    if "snapdragon 8 gen 1" in p:
        return "Snapdragon 8 (Flagship)"
    if "snapdragon 8s" in p:
        return "Snapdragon 8s (Near-Flagship)"
    if "snapdragon 7" in p:
        return "Snapdragon 7 (Upper-Mid)"
    if "snapdragon 6" in p:
        return "Snapdragon 6 (Mid)"
    if "snapdragon 4" in p or "snapdragon 4" in p:
        return "Snapdragon 4/5 (Entry-Mid)"
    if re.search(r"snapdragon\s*(480|460|450|439|430|425|400)", p):
        return "Snapdragon 4/5 (Entry-Mid)"
    if re.search(r"snapdragon\s*(6[5-9][0-9]|7[0-9][0-9]|7[0-9][0-9]g)", p):
        return "Snapdragon 6/7xx (Mid)"
    if "snapdragon" in p:
        return "Snapdragon (Other)"
    if "dimensity 9" in p:
        return "Dimensity 9xxx (Flagship)"
    if "dimensity 8" in p:
        return "Dimensity 8xxx (Upper-Mid)"
    if "dimensity 7" in p:
        return "Dimensity 7xxx (Mid)"
    if "dimensity 6" in p or "dimensity 1" in p or "dimensity 800" in p or "dimensity 900" in p:
        return "Dimensity 6xx-1xxx (Entry-Mid)"
    if "dimensity" in p:
        return "Dimensity (Other)"
    if "helio g9" in p:
        return "Helio G9x (Mid-Budget)"
    if "helio g8" in p or "helio g7" in p:
        return "Helio G7-G8x (Budget)"
    if "helio g" in p:
        return "Helio G (Budget)"
    if "helio p" in p or "helio a" in p:
        return "Helio P/A (Budget)"
    if "kirin 9000" in p or "kirin 9010" in p:
        return "Kirin 9000 (Flagship)"
    if "kirin 9" in p:
        return "Kirin 9xx (Flagship)"
    if "kirin 8" in p or "kirin 7" in p or "kirin 710" in p or "kirin 820" in p or "kirin 985" in p or "kirin 990" in p:
        return "Kirin 7xx-8xx (Mid)"
    if "kirin" in p:
        return "Kirin (Other)"
    if "exynos 2" in p:
        return "Exynos 2xxx (Flagship)"
    if "exynos 1" in p:
        return "Exynos 1xxx (Mid)"
    if "exynos 9" in p or "exynos 8" in p or "exynos 7" in p:
        return "Exynos 7xx-9xx (Budget-Mid)"
    if "exynos" in p:
        return "Exynos (Other)"
    if "tensor" in p:
        return "Google Tensor"
    if "unisoc" in p or "spreadtrum" in p:
        return "Unisoc (Budget)"
    if "mt8" in p or "mediatek mt" in p:
        return "MediaTek MT (Budget)"
    return "Other"


def assign_price_segment(india_price: float | None, thresholds: dict | None = None) -> str:
    """Assign price segment based on India launch price."""
    if thresholds is None:
        thresholds = {
            "Budget": 15000,
            "Mid-Range": 30000,
            "Upper Mid-Range": 60000,
            "Premium": 100000,
        }
    if india_price is None or np.isnan(india_price):
        return "Unknown"
    if india_price < thresholds["Budget"]:
        return "Budget"
    if india_price < thresholds["Mid-Range"]:
        return "Mid-Range"
    if india_price < thresholds["Upper Mid-Range"]:
        return "Upper Mid-Range"
    if india_price < thresholds["Premium"]:
        return "Premium"
    return "Ultra Premium"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_and_clean(path: Path = RAW_PATH) -> pd.DataFrame:
    """
    Load the raw CSV, apply all cleaning steps, return clean DataFrame.
    Original columns are preserved. New columns are added with _num / _clean suffix.
    """
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")

    # Drop blank trailing row
    df = df.dropna(how="all").reset_index(drop=True)

    # Normalise Company Name capitalisation
    df["Company Name"] = df["Company Name"].str.strip()
    df.loc[df["Company Name"] == "Poco", "Company Name"] = "POCO"

    # --- Numeric feature extraction ---
    df["RAM_num"] = df["RAM"].apply(parse_ram)
    df["Weight_num"] = df["Mobile Weight"].apply(parse_weight)
    df["FrontCam_num"] = df["Front Camera"].apply(parse_camera_primary)
    df["BackCam_primary"] = df["Back Camera"].apply(parse_camera_primary)
    df["BackCam_lenses"] = df["Back Camera"].apply(count_camera_lenses)
    df["Battery_num"] = df["Battery Capacity"].apply(parse_battery)
    df["Screen_num"] = df["Screen Size"].apply(parse_screen)

    # --- Price columns ---
    df["India_Price"] = df["Launched Price (India)"].apply(parse_india_price)
    df["Pakistan_Price"] = df["Launched Price (Pakistan)"].apply(parse_pakistan_price)
    df["China_Price"] = df["Launched Price (China)"].apply(parse_china_price)
    df["USA_Price"] = df["Launched Price (USA)"].apply(parse_usa_price)
    df["Dubai_Price"] = df["Launched Price (Dubai)"].apply(parse_dubai_price)

    # --- Derived flags ---
    df["Is_Tablet"] = df.apply(
        lambda r: is_tablet(r["Model Name"], r["Screen_num"]), axis=1
    )
    df["Is_Foldable"] = df["Model Name"].apply(is_foldable)

    # --- Processor family grouping ---
    df["Processor_Family"] = df["Processor"].apply(assign_processor_family)

    # --- Price segment ---
    df["Price_Segment"] = df["India_Price"].apply(assign_price_segment)

    # --- Deduplication (keep first occurrence) ---
    df["_dup_key"] = df["Company Name"] + "|" + df["Model Name"]
    duplicates_mask = df.duplicated(subset="_dup_key", keep="first")
    n_dupes = duplicates_mask.sum()
    df = df[~duplicates_mask].copy()
    df.drop(columns=["_dup_key"], inplace=True)
    df.attrs["n_duplicates_removed"] = int(n_dupes)

    df = df.reset_index(drop=True)
    return df


def get_clean_df(force_reload: bool = False) -> pd.DataFrame:
    """Return cached cleaned DataFrame; reload if missing or forced."""
    if not force_reload and PROCESSED_PATH.exists():
        return pd.read_csv(PROCESSED_PATH)
    df = load_and_clean()
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    return df


if __name__ == "__main__":
    df = load_and_clean()
    print(f"Shape after cleaning: {df.shape}")
    print(f"Duplicates removed: {df.attrs.get('n_duplicates_removed', 0)}")
    print(df[["Company Name", "Model Name", "India_Price", "RAM_num", "Battery_num", "Price_Segment"]].head(10))
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved to {PROCESSED_PATH}")
