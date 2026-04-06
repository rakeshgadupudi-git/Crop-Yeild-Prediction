# Hybrid Crop Yield Prediction using NDVI Time Series and SAR Backscatter with Weather Data

A machine learning system that predicts **crop yield and production** for sugarcane and rice in Maharashtra, India, by combining satellite remote sensing with meteorological data. The project follows a hybrid, two-stage approach:

1. **Stage 1 — NDVI-based pipeline** (Phases 1–3): Uses optical NDVI from Sentinel-2 with NASA POWER weather data and classical regression models.
2. **Stage 2 — SAR-based extension**: Replaces NDVI with Sentinel-1 microwave backscatter (VH, VV, RVI) to enable cloud-penetrating, monsoon-season monitoring.

> **Study Region:** Vidani village, Phaltan Tehsil, Satara District, Maharashtra, India (17.9824°N, 74.5113°E)
> **Crops:** Sugarcane (Satara district) · Rice (Maharashtra state-wide)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [What is NDVI?](#what-is-ndvi)
- [What is SAR?](#what-is-sar)
- [Project Architecture](#project-architecture)
- [Folder Structure](#folder-structure)
- [Data Sources](#data-sources)
- [Pipeline Overview](#pipeline-overview)
  - [Stage 1: NDVI-based Pipeline (Phase 1–3)](#stage-1-ndvi-based-pipeline-phase-13)
  - [Stage 2: SAR-based Pipeline](#stage-2-sar-based-pipeline)
- [Feature Engineering](#feature-engineering)
- [Models and Results](#models-and-results)
  - [NDVI Phase 3 Results](#ndvi-phase-3-results)
  - [SAR Pipeline Results](#sar-pipeline-results)
- [NDVI vs SAR Comparison](#ndvi-vs-sar-comparison)
- [Limitations](#limitations)
- [Software Requirements](#software-requirements)
- [How to Run](#how-to-run)

---

## Problem Statement

Previous research relied on either weather parameters **or** remote sensing data for crop yield prediction — rarely both together. Coarser-resolution satellites (MODIS, Landsat-7/8) were suitable for state/country scale but inadequate for regional farmland analysis.

This project proposes a **novel hybrid approach** combining:
- **Weather parameters** from NASA POWER (13 agro-meteorological variables)
- **NDVI time series** from Sentinel-2 (10 m resolution, 5-day revisit)
- **SAR backscatter** from Sentinel-1 (10 m, 6-day revisit, cloud-transparent)

The key challenge is that NDVI fails during the Indian monsoon season (June–September) due to persistent cloud cover — exactly when crops are in their most critical growth stages. The SAR extension addresses this by using microwave radar signals that penetrate clouds and operate day and night.

---

## What is NDVI?

The **Normalized Difference Vegetation Index (NDVI)** is a spectral index derived from multispectral satellite imagery that quantifies vegetation density and health.

```
NDVI = (NIR − Red) / (NIR + Red)
```

| Band | Sentinel-2 | Sensitivity |
|------|-----------|-------------|
| Red | B4 | Chlorophyll absorption |
| NIR | B8 | Leaf cell structure reflection |

- Range: **−1 to +1**
- Positive values → dense, healthy green vegetation
- Near-zero → bare soil, sparse cover
- Negative → water, ice, clouds

Sentinel-2 provides NDVI observations every 5 days at 10 m resolution, enabling monthly time series analysis over agricultural regions.

---

## What is SAR?

**Synthetic Aperture Radar (SAR)** is an active remote sensing technology that transmits its own microwave signal and measures the backscatter energy from the Earth's surface. Unlike optical sensors, SAR:
- Penetrates clouds and rain
- Operates at night
- Measures physical structure (biomass, roughness, moisture) rather than spectral reflectance

Sentinel-1 operates in **C-band** (5.6 cm wavelength) with two polarization channels:

| Channel | Description | Crop Sensitivity |
|---------|-------------|-----------------|
| VV | Vertical transmit, Vertical receive | Soil moisture, surface roughness |
| VH | Vertical transmit, Horizontal receive | Vegetation volume scattering, biomass |

### Radar Vegetation Index (RVI)

Analogous to NDVI in the microwave domain:

```
RVI = (4 × VH_linear) / (VV_linear + VH_linear)
```

where `linear = 10^(dB/10)`

- Range: **0 (bare soil) to 1 (dense crop)**
- Values > 0.6 indicate a well-developed crop canopy
- Especially sensitive to tall crops like sugarcane

---

## Project Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    HYBRID PREDICTION SYSTEM                      │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │          STAGE 1: NDVI-BASED PIPELINE (Phases 1–3)         │ │
│  │                                                             │ │
│  │  Phase 1            Phase 2           Phase 3              │ │
│  │  Weather Data   →   NDVI Extraction → Yield Prediction     │ │
│  │  (NASA POWER)       (Sentinel-2)      (Random Forest)      │ │
│  │  2000–2020          2016–2019         R² = 0.92            │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │           STAGE 2: SAR-BASED EXTENSION                     │ │
│  │                                                             │ │
│  │  SAR Extraction  →  Dataset Assembly  →  Model Training    │ │
│  │  (Sentinel-1)       (Weather + SAR)      (HistGBM + CV)    │ │
│  │  2015–2025          Annual features      Real-time predict  │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
project-root/
│
├── NDVI_Based_Analysis/             # Optical pipeline: weather, NDVI, and yield prediction
│   ├── preprocessing/               # Weather data collection and time series analysis
│   │   ├── notebooks/               # 12 Jupyter notebooks (one per weather parameter)
│   │   │   ├── AverageTemperature.ipynb
│   │   │   ├── MaxTemperature.ipynb
│   │   │   ├── MinTemperature.ipynb
│   │   │   ├── Precipitation.ipynb
│   │   │   ├── RelativeHumidity.ipynb
│   │   │   ├── SpecificHumidity.ipynb
│   │   │   ├── Dewpoint.ipynb
│   │   │   ├── CloudCover.ipynb
│   │   │   ├── MaxWind.ipynb
│   │   │   ├── MinWind.ipynb
│   │   │   ├── surfacepressure.ipynb
│   │   │   └── CorelationMatrix.ipynb
│   │   ├── csv/                     # Per-parameter monthly data with 2020 forecasts
│   │   │   ├── AverageTemperature/
│   │   │   ├── MaxTemperature/
│   │   │   ├── Precipitation/
│   │   │   └── ... (9 more)
│   │   ├── Weather_Dataset.csv      # Combined monthly weather data (2000–2020)
│   │   └── Phase1_Output.csv        # Annual aggregated weather features
│   │
│   ├── NDVI/                        # NDVI time series processing
│   │   ├── notebooks/
│   │   │   └── NDVI Prediction.ipynb  # GEE-based NDVI extraction and merging
│   │   └── csv/
│   │       ├── ndvi.csv             # Raw monthly NDVI (2016–2019)
│   │       ├── 2020ndvi.csv         # Forecasted 2020 NDVI
│   │       └── ndvi_with_weather.csv  # Combined NDVI + weather features
│   │
│   └── Crop_Pred/                   # Yield and production prediction models
│       ├── notebooks/ndvi/
│       │   ├── Final_Regression_File.ipynb   # EDA + baseline model validation on Maharashtra rice data
│       │   ├── Yield_Prediction.ipynb        # Actual sugarcane yield prediction (train/test → 2020 forecast)
│       │   └── Production_Prediction.ipynb   # Actual sugarcane production prediction (train/test → 2020 forecast)
│       └── csv/ndvi/
│           ├── phase3_input_yield.csv        # Historical yield training data (weather + soil + NDVI + Yield)
│           ├── phase3_input_prod.csv         # Historical production training data (weather + soil + NDVI + Production)
│           ├── 2020prediction.csv            # 2020 input features for final forecast
│           └── Production_with_weather_ndvi.csv  # Combined reference dataset (1999–2025)
│
├── SAR_Yield/                       # SAR-based alternative pipeline
│   ├── scripts/
│   │   ├── sar_extraction_gee.py    # Extract monthly VH/VV/RVI from Sentinel-1
│   │   ├── prepare_sar_dataset.py   # Assemble weather + SAR + yield targets
│   │   ├── train_model.py           # Train HistGBM with TimeSeriesSplit + GridSearchCV
│   │   └── predict_realtime.py      # Real-time current-season prediction
│   ├── notebooks/
│   │   ├── SAR_Yeild.ipynb          # EDA, training, evaluation for yield
│   │   └── SAR_Prod.ipynb           # EDA, training, evaluation for production
│   ├── csv/
│   │   ├── sar_timeseries.csv       # Monthly Sentinel-1 backscatter (2015–2025)
│   │   ├── sar_with_weather.csv     # Combined monthly weather + SAR
│   │   ├── input_prod_sar.csv       # Annual training data (production target)
│   │   ├── phase3_input_yield_sar.csv  # Annual training data (yield target)
│   │   └── 2026prediction_sar.csv   # Input row for 2026 forecast
│   └── models/
│       ├── best_yield_model.pkl     # Trained HistGradientBoostingRegressor
│       ├── scaler.pkl               # Fitted StandardScaler
│       └── imputer.pkl              # Fitted SimpleImputer
│
├── Dataset/                         # Raw authoritative source data
│   ├── NDVI/
│   │   └── 2000-2019-L7.csv         # Landsat-7 NDVI time series
│   ├── Weather_Dataset.xlsx         # Monthly weather 2000–2018
│   ├── Soil_Dataset.xlsx            # Soil nutrient quality codes
│   ├── Dt_production.xlsx           # Historical crop production records
│   ├── Sugarcane_Production_Satara.xlsx
│   ├── maharashtra_Rice_CropR1.csv  # Rice across all 34 districts (1997–2019)
│   └── Rainfall_2000_2020.csv
│
├── data_ingestion/
│   └── weather_api.py               # NASA POWER API fetch module
│
├── results/                         # Model outputs and visualizations
│   ├── correlation_matrix.png
│   ├── actual_vs_predicted.png
│   ├── feature_importance.png
│   ├── sar_distributions.png
│   ├── year_comparison.png
│   ├── output.txt
│   └── regression_output.txt
│
├── logs/                            # Execution logs
├── README.md                        # This file (NDVI + SAR hybrid overview)
└── sar.md                           # Detailed SAR module documentation
```

---

## Data Sources

| Source | Data | Period | Resolution |
|--------|------|--------|-----------|
| NASA POWER API | 11 weather parameters (temp, precipitation, humidity, wind, pressure, cloud cover) | 2000–2025 | Monthly |
| Google Earth Engine — Sentinel-2 | NDVI (B4 red + B8 NIR) | 2016–2020 | 10 m, 5-day revisit |
| Google Earth Engine — Sentinel-1 GRD | VH backscatter, VV backscatter, RVI | 2015–2025 | 10 m, 6-day revisit |
| Ministry of Agriculture & Farmers Welfare | Crop yield (t/ha) and production (tonnes) | 1997–2022 (official) | District level |
| NASA CHRS Data Portal | Precipitation cross-validation | 2000–2020 | Monthly |
| Field surveys + Government records | Soil quality (N, P, K, pH) | Static | Region constant |

### Weather Parameters (NASA POWER — 11 variables)

| Parameter ID | Description | Unit |
|-------------|-------------|------|
| PRECTOTCORR | Total Precipitation | mm/month |
| T2M_MAX | Maximum Temperature | °C |
| T2M_MIN | Minimum Temperature | °C |
| T2M | Average Temperature | °C |
| RH2M | Relative Humidity | % |
| T2MDEW | Dew Point Temperature | °C |
| QV2M | Specific Humidity | kg/kg |
| WS2M_MAX | Maximum Wind Speed | m/s |
| WS2M_MIN | Minimum Wind Speed | m/s |
| PS | Surface Pressure | kPa |
| CLOUD_AMT | Cloud Amount | fraction |

---

## Pipeline Overview

### Stage 1: NDVI-based Pipeline (Phase 1–3)

#### Phase 1 — Weather Data Collection and Time Series Forecasting

```
Input : NASA POWER API monthly data (2000–2020)
         Satara coordinates: 17.9824°N, 74.5113°E
         ↓
Process: 12 separate notebooks — one per weather parameter
         - Load and visualise monthly trends (2000–2019)
         - Forecast 2020 value using time series methods
         - Compute correlation matrix across all parameters
         ↓
Output : Phase1_Output.csv
         Annual aggregated features:
           • Precipitation (sum)  • MaxTemp, MinTemp (max/min)
           • AvgTemp, Humidity, Dew Point, Pressure (mean)
           • Max/Min Wind Speed (max/min)
```

#### Phase 2 — NDVI Time Series Processing

```
Input : Sentinel-2 satellite imagery via Google Earth Engine
        Bands: B4 (Red), B8 (NIR)
        ↓
Process: NDVI Prediction.ipynb
         - Compute NDVI = (NIR - Red) / (NIR + Red) per observation
         - Monthly median composites to reduce noise (2016–2019)
         - Aggregate to annual mean per region
         ↓
Merge  : Combine NDVI time series with Phase 1 weather outputs
         ↓
Output : ndvi_with_weather.csv (combined monthly/annual feature matrix)
```

#### Phase 3 — Yield and Production Prediction

This phase contains three notebooks with distinct roles:

**A. Exploratory Data Analysis & Baseline Validation — `Final_Regression_File.ipynb`**

```
Input : Dataset/maharashtra_Rice_CropR1.csv
        926 records, Maharashtra rice crop (1997–2019)
        Features: District_Name, Crop_Year, Season, Area,
                  Rainfall, MaxTemp → Productivity (t/ha)
        ↓
Process: - Visualise production, rainfall, temperature trends by year
         - Correlation matrix across features
         - Encode District_Name (LabelEncoder + OneHotEncoder), Season
         - 90% train / 10% test split
         - Train and compare 3 baseline regression models:
             • Decision Tree Regressor     R² = 0.79
             • KNeighbors (n=3)            R² = 0.946
             • Random Forest (1000 trees)  R² = 0.68
        ↓
Purpose: EDA and baseline validation only — not used for sugarcane forecasting.
         Demonstrates that classical ML can model rice productivity
         across Maharashtra districts with high accuracy.
```

**B. Sugarcane Yield Prediction — `Yield_Prediction.ipynb`**

```
Input : Crop_Pred/csv/ndvi/phase3_input_yield.csv
        Historical annual data — weather + soil (N,P,K,pH) + NDVI + Yield
        ↓
Process: - Correlation analysis across all features
         - 90% train / 10% test split
         - Train and compare models (Decision Tree, Random Forest)
         - Best model: Random Forest (1000 trees) → R² = 0.92
         ↓
Forecast: Apply best model to 2020prediction.csv
Output  : Sugarcane Yield (2020) = 94.55 tonnes/hectare
```

**C. Sugarcane Production Prediction — `Production_Prediction.ipynb`**

```
Input : Crop_Pred/csv/ndvi/phase3_input_prod.csv
        Historical annual data — weather + soil (N,P,K,pH) + NDVI + Production
        ↓
Process: - Correlation analysis across all features
         - 90% train / 10% test split
         - Train and compare models (Decision Tree, Random Forest)
         - Best model: Random Forest (1000 trees) → R² = 0.92
         ↓
Forecast: Apply best model to 2020prediction.csv
Output  : Sugarcane Production (2020, Vedhani) = 1,431,738 tonnes
```

---

### Stage 2: SAR-based Pipeline

The SAR pipeline overcomes the cloud cover limitation of optical NDVI by using microwave backscatter from Sentinel-1, which operates through clouds and at night.

```
Step 1 — SAR Extraction (sar_extraction_gee.py)
   Authenticate GEE
   Filter Sentinel-1 IW DESCENDING collection (C-band, 2015–2025)
   Compute monthly mean VH, VV per region (5 km buffer around Vidani)
   Derive RVI = 4×VH_linear / (VV_linear + VH_linear)
   → Output: sar_timeseries.csv (monthly, 2015–2025)

Step 2 — Dataset Assembly (prepare_sar_dataset.py)
   Fetch annual NASA POWER weather (11 parameters)
   Aggregate monthly SAR to annual means
   Append soil quality constants (N, P, K, pH)
   Append official yield/production targets (2015–2022) + estimates (2023–2025)
   → Output: input_prod_sar.csv, phase3_input_yield_sar.csv

Step 3 — Model Training (train_model.py)
   Feature engineering:
     - Yield_lag1: previous year yield (temporal dependency)
     - Precip_rolling3: 3-year rolling mean precipitation
   Preprocessing: SimpleImputer (median) → StandardScaler
   Cross-validation: TimeSeriesSplit (n=3) — respects temporal order
   GridSearchCV over HistGradientBoostingRegressor:
     - learning_rate: [0.01, 0.05, 0.1]
     - max_depth:     [2, 3, 5]
     - max_iter:      [50, 100, 200]
   → Output: best_yield_model.pkl, scaler.pkl, imputer.pkl

Step 4 — Real-Time Prediction (predict_realtime.py)
   Fetch current-year weather from NASA POWER
   Fetch latest month's SAR from GEE
   Engineer lag and rolling features
   Apply imputer → scaler → saved model
   → Output: Predicted yield for current season (tonnes/ha)
```

---

## Feature Engineering

### NDVI Pipeline — Features by Notebook

**EDA Baseline — `Final_Regression_File.ipynb`** (input: `maharashtra_Rice_CropR1.csv`)

| Category | Features | Count |
|----------|---------|-------|
| District (one-hot) | 34 Maharashtra districts | 34 |
| Season | Encoded categorical | 1 |
| Climate | Rainfall, MaxTemp | 2 |
| Agricultural | Area under cultivation | 1 |
| **Targets** | Production (tonnes), Productivity (t/ha) | 2 |

Total: **926 records** (1997–2019, Maharashtra rice across all districts) — used for EDA and baseline validation only.

**Actual Prediction — `Yield_Prediction.ipynb` / `Production_Prediction.ipynb`** (input: `phase3_input_yield.csv` / `phase3_input_prod.csv`)

| Category | Features | Count |
|----------|---------|-------|
| Weather | Precipitation, MaxTemp, MinTemp, AvgTemp, SpecificHumidity, RelativeHumidity, SurfacePressure, DewPoints, MinWind, MaxWind, CloudCoverage | 11 |
| Soil | Nitrogen, Phosphorus, Potassium, pH | 4 |
| Vegetation | NDVI | 1 |
| **Target** | Yield (t/ha) or Production (tonnes) | 1 |

Total: Annual records 1999–2019, Satara sugarcane.

### SAR Pipeline — Training Features

| Category | Features | Count |
|----------|---------|-------|
| Weather | Precipitation, MaxTemp, MinTemp, AvgTemp, SpecificHumidity, RelativeHumidity, SurfacePressure, DewPoint, MinWind, MaxWind, CloudCoverage | 11 |
| Soil Quality | Nitrogen, Phosphorus, Potassium, pH (ordinal codes) | 4 |
| SAR Backscatter | VH_mean, VV_mean, RVI | 3 |
| Engineered | Yield_lag1 (previous year), Precip_rolling3 (3-yr rolling avg) | 2 |
| **Target** | Yield (t/ha) or Production (tonnes) | 1 |

Total training samples: **11 annual records** (2015–2025)

### SAR Backscatter Typical Ranges

| Parameter | Typical Range | Interpretation |
|-----------|-------------|----------------|
| VH_mean | −19.3 to −13.2 dB | Cross-pol; higher = denser canopy |
| VV_mean | −12.5 to −7.5 dB | Co-pol; sensitive to soil moisture |
| RVI | 0.54 to 0.77 | > 0.6 = well-developed crop canopy |

---

## Models and Results

### NDVI Phase 3 Results

#### EDA & Baseline — `Final_Regression_File.ipynb` (Maharashtra Rice, 926 records)

Input: `Dataset/maharashtra_Rice_CropR1.csv` — 926 district-year records (1997–2019), 90/10 split.
Purpose: Exploratory analysis and baseline model validation only.

| Model | R² |
|-------|----|
| K-Nearest Neighbors (n=3) | 0.946 |
| Decision Tree Regressor | 0.790 |
| Random Forest (1000 trees) | 0.684 |

> This notebook is EDA and baseline validation. It is not used for sugarcane yield forecasting.

---

#### Actual Prediction — `Yield_Prediction.ipynb` & `Production_Prediction.ipynb` (Sugarcane, Satara)

Input: `phase3_input_yield.csv` / `phase3_input_prod.csv` — weather + soil (N,P,K,pH) + NDVI, 90/10 split.

| Task | Best Model | R² |
|------|-----------|-----|
| Sugarcane Yield (t/ha) | Random Forest (1000 trees) | **0.92** |
| Sugarcane Production (tonnes) | Random Forest (1000 trees) | **0.92** |

**2020 Forecast Results:**

| Target | Predicted Value |
|--------|----------------|
| Yield | 94.55 tonnes/hectare |
| Production (Vedhani) | 1,431,738 tonnes |

**Best model: Random Forest (1000 estimators)** — R² of 0.92 means the model explains 92% of the variance in sugarcane yield and production.

### SAR Pipeline Results

Training set: 11 annual records (Satara sugarcane, 2015–2025)
Cross-validation: Leave-One-Out CV (maximises training data on small dataset)

#### Production Prediction (LOO-CV)

| Model | MAE (tonnes) | RMSE (tonnes) | R² |
|-------|------------|-------------|-----|
| Random Forest | 116,542 | 196,617 | −0.01 |
| HistGradientBoosting | 145,242 | 214,704 | −0.21 |
| SVR | 148,098 | 207,344 | −0.13 |
| KNN | 161,249 | 205,246 | −0.11 |
| Linear Regression | 363,201 | 483,514 | −5.14 |

> Negative R² values are expected with only 11 training samples. Random Forest is most robust across folds.

#### 2026 Production Forecasts

| Model | Predicted Production (tonnes) |
|-------|------------------------------|
| Random Forest | 1,617,085 |
| Linear Regression | 1,617,103 |
| KNN | 1,594,500 |
| HistGradientBoosting | 1,477,512 |
| SVR | 1,462,600 |

---

## NDVI vs SAR Comparison

| Aspect | NDVI (Sentinel-2, Optical) | SAR (Sentinel-1, Microwave) |
|--------|--------------------------|--------------------------|
| Cloud Penetration | No — affected by clouds | Yes — cloud-transparent |
| Night Acquisition | No | Yes |
| Monsoon Season Use | Limited (June–Sept) | Continuous year-round |
| Vegetation Sensitivity | Surface chlorophyll | Volume scattering / biomass |
| Soil Moisture Detection | Indirect | Direct (VV channel) |
| Revisit Cycle | 5 days | 6 days |
| Spatial Resolution | 10 m | 10 m |
| Index Used | NDVI | RVI, VH, VV |
| Training Samples | 926 (rice EDA) / sugarcane annual (Satara) | 11 (sugarcane, Satara) |
| Best Model R² | 0.92 (Random Forest, sugarcane prediction) | Negative (insufficient data) |
| Prediction Horizon | Seasonal | 20+ days pre-harvest |

---


## Software Requirements

| Tool / Library | Version | Purpose |
|---------------|---------|---------|
| Python | 3.7+ | Core language |
| Jupyter Notebook | 6.0+ | Interactive data exploration |
| Google Earth Engine (`earthengine-api`) | Latest | Sentinel-1 and Sentinel-2 data extraction |
| scikit-learn | Latest | Model training, evaluation, preprocessing |
| pandas / numpy | Latest | Data manipulation and feature engineering |
| matplotlib / seaborn | Latest | Visualizations |
| joblib | Latest | Model artifact serialization |
| QGIS | 3.10.5 | GIS mapping of Maharashtra shapefiles |
| Apache Tomcat | 9.0 | Web GIS portal hosting |
| Geoserver | 2.17 | Shapefile hosting and visualization |
| PostgreSQL + PostGIS | 9.1 / 2.2 | Geospatial database for coordinates and attributes |

---

## How to Run

### Stage 1: NDVI-based Pipeline

**Phase 1 — Weather Data**
Open and run each notebook in `NDVI_Based_Analysis/preprocessing/notebooks/` sequentially. Each notebook handles one weather parameter and outputs a CSV to `NDVI_Based_Analysis/preprocessing/csv/`.

**Phase 2 — NDVI Extraction**
Open `NDVI_Based_Analysis/NDVI/notebooks/NDVI Prediction.ipynb` in Google Colab or Jupyter with GEE authentication. Run cells to extract NDVI from Sentinel-2 and merge with preprocessing output.

**Phase 3 — Yield Prediction**
```bash
# Open in Jupyter or Colab
NDVI_Based_Analysis/Crop_Pred/notebooks/ndvi/Final_Regression_File.ipynb
```
Run all cells to train Decision Tree, KNN, and Random Forest models and evaluate performance.

---

### Stage 2: SAR-based Pipeline

**Step 1 — Extract SAR Time Series** (requires GEE authentication)
```bash
earthengine authenticate
python SAR_Yield/scripts/sar_extraction_gee.py
```
Outputs `SAR_Yield/csv/sar_timeseries.csv` with monthly Sentinel-1 backscatter.

**Step 2 — Prepare Training Dataset**
```bash
python SAR_Yield/scripts/prepare_sar_dataset.py
```
Fetches NASA POWER weather and merges with SAR data. Outputs annual training CSVs.

**Step 3 — Train Model**
```bash
python SAR_Yield/scripts/train_model.py
```
Runs GridSearchCV with TimeSeriesSplit. Saves trained model artifacts to `SAR_Yield/models/`.

**Step 4 — Real-Time Prediction**
```bash
python SAR_Yield/scripts/predict_realtime.py
```
Fetches current-season weather and SAR data, applies the saved model, and prints the predicted yield.

**Notebooks (interactive EDA)**
```
SAR_Yield/notebooks/SAR_Yeild.ipynb   # Yield: EDA, model comparison, evaluation
SAR_Yield/notebooks/SAR_Prod.ipynb    # Production: EDA, model comparison, 2026 forecasts
```

---

## Project Statistics

| Metric | Value |
|--------|-------|
| Geographic focus | Satara District + Maharashtra state |
| Primary crops | Sugarcane (Satara), Rice (Maharashtra-wide) |
| Study period | 1997–2025 |
| Jupyter notebooks | 19 total (12 preprocessing + 1 NDVI + 3 Crop_Pred + 2 SAR + 1 correlation) |
| Python scripts | 5 (weather_api.py + 4 SAR scripts) |
| Weather parameters | 11 (NASA POWER) |
| NDVI training samples | 926 (rice, district-level) |
| SAR training samples | 11 (sugarcane, annual) |
| Best NDVI model R² (sugarcane prediction) | 0.92 (Random Forest) |
| 2026 SAR production forecast | ~1.48–1.62 million tonnes |
