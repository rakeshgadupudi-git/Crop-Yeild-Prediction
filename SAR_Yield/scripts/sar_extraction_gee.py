"""
SAR Time Series Extraction from Sentinel-1 using Google Earth Engine
====================================================================
This script extracts monthly VH, VV backscatter and derived indices (VH/VV ratio, RVI)
for the Satara district (Vidani village region) in Maharashtra, India.

Region: Latitude 17.9824°N, Longitude 74.5113°E (Phaltan Tehsil, Satara, Maharashtra)
Satellite: Sentinel-1 GRD (Ground Range Detected), IW mode
Time Range: 2015-01 to 2025-12

Usage:
    1. Install: pip install earthengine-api pandas
    2. Authenticate: earthengine authenticate
    3. Run: python sar_extraction_gee.py

Output: ../csv/sar_timeseries.csv
"""

import ee
import pandas as pd
import os

# =====================================================================
# CONFIGURATION
# =====================================================================

# Initialize Earth Engine with Project ID
PROJECT_ID = 'gen-lang-client-0137993599'
ee.Initialize(project=PROJECT_ID)

# Study area: Vidani village, Phaltan tehsil, Satara district, Maharashtra
# Using a 5km buffer around the village center for regional aggregation
LATITUDE = 17.9824
LONGITUDE = 74.5113
BUFFER_RADIUS = 5000  # meters

# Time range
START_YEAR = 2015
END_YEAR = 2025

# Output path
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'csv')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'sar_timeseries.csv')


# =====================================================================
# FUNCTIONS
# =====================================================================

def get_sentinel1_monthly(roi, year, month):
    """
    Get monthly mean Sentinel-1 SAR backscatter for a region of interest.
    
    Parameters:
        roi: ee.Geometry - Region of interest
        year: int - Year
        month: int - Month (1-12)
    
    Returns:
        dict with VH_mean, VV_mean, VH_VV_ratio, RVI or None if no data
    """
    start_date = ee.Date.fromYMD(year, month, 1)
    end_date = start_date.advance(1, 'month')
    
    # Filter Sentinel-1 GRD collection
    collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                  .filterBounds(roi)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.eq('instrumentMode', 'IW'))
                  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
                  .select(['VV', 'VH']))
    
    count = collection.size().getInfo()
    if count == 0:
        return None
    
    # Compute monthly median composite (median is more robust to outliers)
    composite = collection.median()
    
    # VH and VV are already in dB in GRD products
    # Calculate VH/VV ratio (in dB domain: VH_dB - VV_dB)
    vh_vv_ratio_db = composite.select('VH').subtract(composite.select('VV')).rename('VH_VV_ratio')
    
    # For RVI, convert from dB to linear power first
    # power = 10^(dB/10)
    vh_linear = ee.Image(10).pow(composite.select('VH').divide(10)).rename('VH_linear')
    vv_linear = ee.Image(10).pow(composite.select('VV').divide(10)).rename('VV_linear')
    
    # RVI = (4 * VH) / (VV + VH)
    rvi = vh_linear.multiply(4).divide(vv_linear.add(vh_linear)).rename('RVI')
    
    # Stack all bands
    stacked = composite.addBands(vh_vv_ratio_db).addBands(rvi)
    
    # Reduce region to get mean values
    stats = stacked.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=10,  # Sentinel-1 resolution
        maxPixels=1e9
    ).getInfo()
    
    return {
        'Year': year,
        'Month': month,
        'VH_mean': round(stats.get('VH', -999), 4),
        'VV_mean': round(stats.get('VV', -999), 4),
        'VH_VV_ratio': round(stats.get('VH_VV_ratio', -999), 4),
        'RVI': round(stats.get('RVI', -999), 4)
    }


def get_latest_sar_data(lat, lon, year, month, buffer_radius=BUFFER_RADIUS):
    """
    Fetches the SAR data for a specific precise month and year for real-time prediction.
    """
    point = ee.Geometry.Point([lon, lat])
    roi = point.buffer(buffer_radius)
    return get_sentinel1_monthly(roi, year, month)

def extract_sar_timeseries(lat=LATITUDE, lon=LONGITUDE, start_year=START_YEAR, end_year=END_YEAR, output_file=OUTPUT_FILE):
    """
    Extract full monthly SAR time series from 2015 to 2025.
    """
    # Define region of interest
    point = ee.Geometry.Point([lon, lat])
    roi = point.buffer(BUFFER_RADIUS)
    
    print(f"Extracting SAR data for region: {lat}°N, {lon}°E")
    print(f"Buffer: {BUFFER_RADIUS}m | Period: {start_year}-{end_year}")
    print("-" * 60)
    
    records = []
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            try:
                result = get_sentinel1_monthly(roi, year, month)
                if result:
                    records.append(result)
                    print(f"  ✓ {year}-{month:02d}: VH={result['VH_mean']:.2f} dB, "
                          f"VV={result['VV_mean']:.2f} dB, RVI={result['RVI']:.4f}")
                else:
                    print(f"  ✗ {year}-{month:02d}: No Sentinel-1 data available")
            except Exception as e:
                print(f"  ✗ {year}-{month:02d}: Error - {str(e)}")
    
    # Create DataFrame and save
    df = pd.DataFrame(records)
    out_dir = os.path.dirname(output_file)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\n{'=' * 60}")
    print(f"Saved {len(records)} records to: {output_file}")
    print(f"Columns: {list(df.columns)}")
    
    return df


# =====================================================================
# MAIN
# =====================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  SAR Time Series Extraction - Sentinel-1 GRD")
    print("  Crop: Sugarcane/Rice | Region: Satara, Maharashtra")
    print("=" * 60)
    print()
    
    df = extract_sar_timeseries()
    
    if len(df) > 0:
        print(f"\nSummary Statistics:")
        print(df.describe().to_string())
    else:
        print("\nNo data extracted. Check your GEE authentication and internet connection.")
