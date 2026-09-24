# Data Dictionary — Mobiles Dataset (2025)

All columns in the cleaned analytical dataset. Raw columns are preserved; engineered columns are added.

---

## Original Columns (Raw)

| Column | Type | Unit | Description | Example | Missing | Analytical Purpose |
|---|---|---|---|---|---|---|
| `Company Name` | string | — | Brand/manufacturer name | `Apple` | 0 | Brand segmentation, grouping |
| `Model Name` | string | — | Full model name including storage | `iPhone 16 128GB` | 0 | Device identification |
| `Mobile Weight` | string | grams | Device weight with unit suffix | `174g` | 0 | Spec comparison (cleaned to float) |
| `RAM` | string | GB | RAM size with unit | `6GB` | 0 | Feature for ML, spec analysis |
| `Front Camera` | string | MP | Front camera spec | `12MP`, `Dual 32MP` | 0 | Spec analysis |
| `Back Camera` | string | MP | Rear camera spec (multi-lens) | `50MP + 12MP` | 0 | Spec analysis |
| `Processor` | string | — | Chip/processor name | `A17 Bionic` | 0 | Processor grouping, ML |
| `Battery Capacity` | string | mAh | Battery with unit suffix | `3,600mAh` | 0 | Spec analysis, ML |
| `Screen Size` | string | inches | Display size | `6.1 inches` | 0 | Spec analysis, ML |
| `Launched Price (Pakistan)` | string | PKR | Pakistan launch price | `PKR 224,999` | ~2 | Global comparison |
| `Launched Price (India)` | string | INR | India launch price | `INR 79,999` | 0 | Primary price variable |
| `Launched Price (China)` | string | CNY | China launch price | `CNY 5,799` | ~2 | Global comparison |
| `Launched Price (USA)` | string | USD | USA launch price | `USD 799` | 0 | Global comparison |
| `Launched Price (Dubai)` | string | AED | Dubai launch price | `AED 2,799` | 0 | Global comparison |
| `Launched Year` | integer | year | Year of launch | `2024` | 0 | Trend analysis, ML |

---

## Engineered Columns (Cleaned Analytical)

| Column | Type | Unit | Description | Example | Source | Notes |
|---|---|---|---|---|---|---|
| `RAM_num` | float | GB | Numeric RAM | `6.0` | `RAM` | `6GB → 6.0`; ambiguous `8GB / 12GB → 8.0` |
| `Weight_num` | float | g | Numeric weight | `174.0` | `Mobile Weight` | Strips `g` |
| `FrontCam_num` | float | MP | Primary front camera MP | `12.0` | `Front Camera` | First numeric value; `Dual 32MP → 32.0` |
| `BackCam_primary` | float | MP | Primary back camera MP | `50.0` | `Back Camera` | First numeric value |
| `BackCam_lenses` | integer | count | Number of rear lenses | `2` | `Back Camera` | Count of `+` separators + 1 |
| `Battery_num` | integer | mAh | Numeric battery capacity | `3600` | `Battery Capacity` | Removes `mAh` and commas |
| `Screen_num` | float | inches | Primary screen size | `6.1` | `Screen Size` | First float; foldables use main screen |
| `India_Price` | float | INR | Numeric India launch price | `79999.0` | `Launched Price (India)` | Handles Indian comma format |
| `Pakistan_Price` | float | PKR | Numeric Pakistan price | `224999.0` | `Launched Price (Pakistan)` | NaN for `Not available` |
| `China_Price` | float | CNY | Numeric China price | `5799.0` | `Launched Price (China)` | NaN for encoding errors |
| `USA_Price` | float | USD | Numeric USA price | `799.0` | `Launched Price (USA)` | Handles decimal format variants |
| `Dubai_Price` | float | AED | Numeric Dubai price | `2799.0` | `Launched Price (Dubai)` | |
| `Is_Tablet` | boolean | — | True if device is a tablet | `False` | Model Name + Screen_num | Screen > 9 in OR model contains iPad/Tab/Pad |
| `Is_Foldable` | boolean | — | True if device is foldable | `False` | Model Name | Detects Fold/Flip/Xs 2/Open/Magic V etc. |
| `Processor_Family` | string | — | Grouped processor family | `Snapdragon 8 (Flagship)` | `Processor` | ~15 families for ML encoding |
| `Price_Segment` | string | — | India price segment | `Mid-Range` | `India_Price` | Budget/Mid/Upper Mid/Premium/Ultra Premium |

---

## Price Segment Thresholds

Thresholds are based on approximate percentiles of the India_Price distribution in this dataset:

| Segment | Threshold | Rationale |
|---|---|---|
| Budget | < ₹15,000 | ~P20 |
| Mid-Range | ₹15,000 – ₹30,000 | ~P20–P40 |
| Upper Mid-Range | ₹30,000 – ₹60,000 | ~P40–P65 |
| Premium | ₹60,000 – ₹1,00,000 | ~P65–P85 |
| Ultra Premium | > ₹1,00,000 | ~P85+ |

---

## Processor Family Groupings

| Family Label | Processor Examples |
|---|---|
| Apple | A11 Bionic – A17 Bionic, A12Z |
| Snapdragon 8 (Flagship) | SD 8 Gen 3/2/1, SD 888, SD 870, SD 865, SD 855, SD 845, SD 835 |
| Snapdragon 8s (Near-Flagship) | SD 8s Gen 3 |
| Snapdragon 7 (Upper-Mid) | SD 7 Gen 3/2/1, SD 7s Gen 2 |
| Snapdragon 6/7xx (Mid) | SD 6 Gen 3/1, SD 695, SD 680, SD 660 etc. |
| Snapdragon 4/5 (Entry-Mid) | SD 480, SD 460, SD 450, SD 4 Gen 1 |
| Dimensity 9xxx (Flagship) | Dimensity 9400, 9300, 9200, 9000 |
| Dimensity 8xxx (Upper-Mid) | Dimensity 8350, 8300, 8200, 8100 |
| Dimensity 7xxx (Mid) | Dimensity 7300, 7200, 7050 |
| Dimensity 6xx-1xxx (Entry-Mid) | Dimensity 6100+, 6020, 1300, 1200, 900 |
| Kirin 9000 (Flagship) | Kirin 9000S, Kirin 9010 |
| Kirin 7xx-8xx (Mid) | Kirin 990, 985, 820, 710 |
| Exynos 2xxx (Flagship) | Exynos 2400, 2200 |
| Exynos 1xxx (Mid) | Exynos 1380, 1280 |
| Exynos 7xx-9xx (Budget-Mid) | Exynos 990, 9825, 850 |
| Google Tensor | Google Tensor G1–G4 |
| Helio G9x / G8x / G7x | MediaTek Helio G99, G96, G88, G85 |
| Helio G (Budget) | MediaTek Helio G80, G70, G37 etc. |
| Helio P/A (Budget) | MediaTek Helio P65, P35, P22, A20 |
| Unisoc (Budget) | Unisoc T606, T612, SC9863A |
| Other | Any unclassified processor |
