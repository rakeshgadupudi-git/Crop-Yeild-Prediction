"""
prepare_sar_dataset.py
======================
Rebuilds SAR_Yield/csv/phase3_input_yield_sar.csv and
phase3_input_prod_sar.csv from authoritative sources:

  1. Real NASA POWER weather via data_ingestion/weather_api.py
  2. Real Sentinel-1 annual SAR means via SAR_Yield/csv/sar_timeseries.csv
  3. Hardcoded yield/production dictionaries (known official values)

Run from the repository root:
    python SAR_Yield/scripts/prepare_sar_dataset.py

The script prints a data-quality report for each year showing:
  - Whether weather came from NASA POWER API or is missing
  - Whether SAR came from >=6 months of Sentinel-1 data
  - Whether yield/production is a verified official figure or estimated

KNOWN DATA QUALITY ISSUES (not fixable by this script):
  - 2015 SAR has only 1 month of data (February) → LOW_COVERAGE flag
  - 2016 SAR has only 4 months of data → LOW_COVERAGE flag
  - 2017 and 2019 have identical yield/production values (likely
    missing-data imputation in the original source)
  - Soil columns are ordinal quality codes, NOT real measurements
  - 2023-2025 yield/production are extrapolated estimates
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Make project root importable regardless of where script is run from
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from data_ingestion.weather_api import fetch_weather_data

# =====================================================================
# CONFIGURATION
# =====================================================================

LAT = 17.9824
LON = 74.5113

TRAINING_START_YEAR = 2015
TRAINING_END_YEAR   = 2025

# Years with fewer SAR monthly observations than this are flagged LOW_COVERAGE
MIN_SAR_MONTHS = 6

# =====================================================================
# HARDCODED KNOWN VALUES
# Update these when official Maharashtra district crop statistics
# for 2023+ are published.
#
# source_quality values:
#   'official'       — from verified government/research records
#   'estimated'      — linearly extrapolated; replace when real data available
#   'duplicate_flag' — identical to another year; likely missing-data imputation
# =====================================================================

KNOWN_YIELD = {
    2015: (96.64,        'official'),
    2016: (124.38,       'official'),
    2017: (99.0,         'duplicate_flag'),   # same value as 2019
    2018: (67.29,        'official'),
    2019: (99.0,         'duplicate_flag'),   # same value as 2017
    2020: (95.5,         'official'),
    2021: (100.2,        'official'),
    2022: (103.8,        'official'),
    2023: (106.5,        'estimated'),
    2024: (109.2,        'estimated'),
    2025: (112.0,        'estimated'),
}

KNOWN_PRODUCTION = {
    2015: (1410443.725,  'official'),
    2016: (1815274.444,  'official'),
    2017: (1444922.863,  'duplicate_flag'),
    2018: (982068.374,   'official'),
    2019: (1444922.863,  'duplicate_flag'),
    2020: (1394000.0,    'official'),
    2021: (1462500.0,    'official'),
    2022: (1515000.0,    'official'),
    2023: (1554500.0,    'estimated'),
    2024: (1594000.0,    'estimated'),
    2025: (1635000.0,    'estimated'),
}

# Soil ordinal quality codes — constant for this region.
# NOTE: These are NOT real kg/ha or pH measurements.
# Values represent relative soil quality levels encoded as integers.
# Replace with actual soil test data when available.
SOIL_CONSTANTS = {
    'Nitrogen':   -1,
    'Phosphorus':  1,
    'Pottasium':   2,
    'pH':          2,
}

# Column order to match existing CSV format exactly
COL_ORDER = [
    'Year', 'Precipitation', 'maxtemp', 'mintemp', 'avgtemp',
    'specifichumidity', 'relativehumidity', 'surfacepressure',
    'dewpoints', 'minwindspeed', 'maxwindspeed', 'cloudcoverage',
    'Nitrogen', 'Phosphorus', 'Pottasium', 'pH',
    'VH_mean', 'VV_mean', 'RVI',
]


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def compute_sar_annual_means(sar_csv_path):
    """
    Read sar_timeseries.csv and compute annual means of VH_mean, VV_mean,
    and RVI for each year using the available monthly observations.

    Years with fewer than MIN_SAR_MONTHS observations are flagged as
    LOW_COVERAGE but still included — real sparse data is better than
    fabricated values.

    Returns dict keyed by year:
        {year: {'VH_mean': float, 'VV_mean': float, 'RVI': float,
                'n_months': int, 'coverage': 'OK' | 'LOW_COVERAGE'}}
    """
    df = pd.read_csv(sar_csv_path)
    annual = {}
    for year, grp in df.groupby('Year'):
        n = len(grp)
        annual[int(year)] = {
            'VH_mean':  round(grp['VH_mean'].mean(), 6),
            'VV_mean':  round(grp['VV_mean'].mean(), 6),
            'RVI':      round(grp['RVI'].mean(), 6),
            'n_months': n,
            'coverage': 'OK' if n >= MIN_SAR_MONTHS else 'LOW_COVERAGE',
        }
    return annual


def fetch_weather_safe(lat, lon, start_year, end_year):
    """
    Fetch NASA POWER annual weather. Returns a dict keyed by year on
    success, or an empty dict if the API is unavailable (e.g. data not
    yet published for recent years).
    """
    try:
        df = fetch_weather_data(lat, lon, start_year, end_year)
        if df is not None and len(df) > 0:
            return df.set_index('Year').to_dict(orient='index')
    except Exception as e:
        print(f"  WARNING: NASA POWER API failed: {e}")
    return {}


def build_dataset(target_col):
    """
    Assemble the complete training dataset for target_col ('Yield' or
    'Production'). Returns (DataFrame, quality_report list).
    """
    print(f"\n{'='*60}")
    print(f"Building dataset  →  target: {target_col}")
    print(f"{'='*60}")

    # Step 1: weather
    print(f"\nStep 1: Fetching NASA POWER weather "
          f"({TRAINING_START_YEAR}–{TRAINING_END_YEAR})...")
    weather_by_year = fetch_weather_safe(LAT, LON,
                                         TRAINING_START_YEAR,
                                         TRAINING_END_YEAR)
    if weather_by_year:
        print(f"  Received data for years: {sorted(weather_by_year.keys())}")
    else:
        print("  WARNING: No weather data received — "
              "weather columns will be NaN (imputed at training time).")

    # Step 2: SAR annual means
    print("\nStep 2: Computing SAR annual means from sar_timeseries.csv...")
    sar_csv = os.path.join(ROOT, 'SAR_Yield', 'csv', 'sar_timeseries.csv')
    sar_by_year = compute_sar_annual_means(sar_csv)
    for yr, info in sorted(sar_by_year.items()):
        flag = f"  [{info['coverage']}]" if info['coverage'] != 'OK' else ''
        print(f"  {yr}: n_months={info['n_months']:2d}  "
              f"VH={info['VH_mean']:9.4f}  "
              f"VV={info['VV_mean']:9.4f}  "
              f"RVI={info['RVI']:.4f}{flag}")

    # Step 3: assemble rows
    print(f"\nStep 3: Assembling rows ({TRAINING_START_YEAR}–"
          f"{TRAINING_END_YEAR})...")
    known_targets = KNOWN_YIELD if target_col == 'Yield' else KNOWN_PRODUCTION

    rows = []
    quality_report = []

    for year in range(TRAINING_START_YEAR, TRAINING_END_YEAR + 1):
        row = {'Year': year}

        # Weather
        if year in weather_by_year:
            row.update(weather_by_year[year])
            wx_status = 'NASA_POWER_REAL'
        else:
            for col in ['Precipitation', 'maxtemp', 'mintemp', 'avgtemp',
                        'specifichumidity', 'relativehumidity',
                        'surfacepressure', 'dewpoints', 'minwindspeed',
                        'maxwindspeed', 'cloudcoverage']:
                row[col] = np.nan
            wx_status = 'MISSING'

        # Soil constants
        row.update(SOIL_CONSTANTS)

        # SAR
        if year in sar_by_year:
            row['VH_mean'] = sar_by_year[year]['VH_mean']
            row['VV_mean'] = sar_by_year[year]['VV_mean']
            row['RVI']     = sar_by_year[year]['RVI']
            sar_status = sar_by_year[year]['coverage']
        else:
            row['VH_mean'] = np.nan
            row['VV_mean'] = np.nan
            row['RVI']     = np.nan
            sar_status = 'MISSING'

        # Target
        if year in known_targets:
            target_val, target_quality = known_targets[year]
            row[target_col] = target_val
        else:
            row[target_col] = np.nan
            target_quality = 'MISSING'

        rows.append(row)
        quality_report.append({
            'Year':        year,
            'Weather':     wx_status,
            'SAR':         sar_status,
            target_col:    target_quality,
        })

    df = pd.DataFrame(rows)
    df = df[COL_ORDER + [target_col]]
    return df, quality_report


def print_quality_report(quality, target_col, out_file):
    print(f"\nData Quality Report — {out_file}")
    print("-" * 65)
    header = f"{'Year':>4}  {'Weather':>16}  {'SAR':>14}  {target_col:>16}"
    print(header)
    print("-" * 65)
    for q in quality:
        wx   = q['Weather']
        sar  = q['SAR']
        tgt  = q[target_col]
        warn = ('  ←' if any(w in f"{wx}{sar}{tgt}"
                              for w in ['MISSING', 'LOW', 'estimated',
                                        'duplicate'])
                else '')
        print(f"{q['Year']:>4}  {wx:>16}  {sar:>14}  {tgt:>16}{warn}")

    print("\nKnown limitations (cannot be fixed by this script):")
    print("  • 2015 SAR: 1 month only (Feb) — very sparse")
    print("  • 2016 SAR: 4 months only — sparse")
    print("  • 2017 & 2019 yield/production are identical — "
          "suspected missing-data imputation in source records")
    print("  • Soil columns are ordinal quality codes, not real measurements")
    print("  • 2023–2025 yield/production are extrapolated estimates — "
          "replace when official Maharashtra statistics are published")


# =====================================================================
# MAIN
# =====================================================================

def main():
    out_dir = os.path.join(ROOT, 'SAR_Yield', 'csv')

    for target_col, out_file in [
        ('Yield',      'phase3_input_yield_sar.csv'),
        ('Production', 'phase3_input_prod_sar.csv'),
    ]:
        df, quality = build_dataset(target_col)
        print_quality_report(quality, target_col, out_file)

        out_path = os.path.join(out_dir, out_file)
        df.to_csv(out_path, index=False)
        print(f"\nSaved {len(df)} rows → {out_path}")

    print("\n" + "="*60)
    print("Done. Re-run this script whenever:")
    print("  • New NASA POWER data is available for 2024/2025")
    print("  • New Sentinel-1 data is added to sar_timeseries.csv")
    print("  • Official yield/production figures for 2023+ are published")
    print("="*60)


if __name__ == "__main__":
    main()
