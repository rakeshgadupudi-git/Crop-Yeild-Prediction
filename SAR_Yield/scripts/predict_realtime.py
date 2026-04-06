import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import warnings

warnings.filterwarnings('ignore')

# Add project root to sys path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from data_ingestion.weather_api import fetch_weather_data
from SAR_Yield.scripts.sar_extraction_gee import get_latest_sar_data

def predict_current_season(lat=17.9824, lon=74.5113):
    current_year = datetime.now().year
    
    print("=== Step 1: Fetching Real-Time Weather Data ===")
    try:
        df_weather = fetch_weather_data(lat, lon, start_year=current_year, end_year=current_year)
    except Exception as e:
        print(f"Note: NASA API returned error for {current_year} (Likely not available yet). Falling back to {current_year - 1}.")
        df_weather = pd.DataFrame() # Trigger fallback
        
    if len(df_weather) == 0:
        current_year = current_year - 1
        df_weather = fetch_weather_data(lat, lon, start_year=current_year, end_year=current_year)

    print("\n=== Step 2: Fetching Real-Time Sentinel-1 SAR Data ===")
    # Fetching SAR data for peak growing season (e.g., September = month 9)
    try:
        sar_data = get_latest_sar_data(lat, lon, year=current_year, month=9)
    except Exception as e:
        print(f"Error fetching SAR data: {e}")
        sar_data = None
        
    if not sar_data:
        print("SAR data not available yet for the specified period. Using Historical averages.")
        sar_data = {'VH_mean': -18.0, 'VV_mean': -10.0, 'RVI': 0.65}
        
    print("\n=== Step 3: Processing & Engineering Features ===")
    # Load the historical dataset to calculate lags and rolling means
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'csv', 'phase3_input_yield_sar.csv')
    df_hist = pd.read_csv(csv_path)
    
    # Create the current year's row
    current_row = {
        'Year': current_year,
        'Precipitation': df_weather['Precipitation'].values[0],
        'maxtemp': df_weather['maxtemp'].values[0],
        'mintemp': df_weather['mintemp'].values[0],
        'avgtemp': df_weather['avgtemp'].values[0],
        'relativehumidity': df_weather['relativehumidity'].values[0],
        'surfacepressure': df_weather['surfacepressure'].values[0],
        'dewpoints': df_weather['dewpoints'].values[0],
        'minwindspeed': df_weather['minwindspeed'].values[0],
        'maxwindspeed': df_weather['maxwindspeed'].values[0],
        'specifichumidity': df_weather['specifichumidity'].values[0],
        'cloudcoverage': df_weather['cloudcoverage'].values[0],
        'Nitrogen': -1,   # Regional Constants
        'Phosphorus': 1,
        'Pottasium': 2,
        'pH': 2,
        'VH_mean': sar_data['VH_mean'],
        'VV_mean': sar_data['VV_mean'],
        'RVI': sar_data['RVI'],
        'Yield': np.nan # To be predicted
    }
    
    df_combined = pd.concat([df_hist, pd.DataFrame([current_row])], ignore_index=True)
    
    # Generate lag and rolling features
    df_combined = df_combined.sort_values('Year').reset_index(drop=True)
    df_combined['Yield_lag1'] = df_combined['Yield'].shift(1)
    df_combined['Yield_lag1'] = df_combined['Yield_lag1'].bfill() # Edge case for first row
    
    df_combined['Precip_rolling3'] = df_combined['Precipitation'].rolling(window=3, min_periods=1).mean()
    df_combined['Temp_rolling3'] = df_combined['avgtemp'].rolling(window=3, min_periods=1).mean()
    
    # Extract only the current year row
    df_target = df_combined[df_combined['Year'] == current_year].copy()
    
    features = [
        'Precipitation', 'maxtemp', 'mintemp', 'avgtemp',
        'specifichumidity', 'relativehumidity', 'surfacepressure',
        'dewpoints', 'minwindspeed', 'maxwindspeed', 'cloudcoverage',
        'Nitrogen', 'Phosphorus', 'Pottasium', 'pH',
        'VH_mean', 'VV_mean', 'RVI', 
        'Yield_lag1', 'Precip_rolling3', 'Temp_rolling3'
    ]
    
    X_target = df_target[features]
    
    print("\n=== Step 4: Loading Model & Predicting ===")
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    model_path = os.path.join(model_dir, 'best_yield_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    imputer_path = os.path.join(model_dir, 'imputer.pkl')
    
    if not os.path.exists(model_path):
        print("Model not found! Please run train_model.py first.")
        return
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    imputer = joblib.load(imputer_path)
    
    X_imputed = imputer.transform(X_target)
    X_scaled = scaler.transform(X_imputed)
    X_scaled = pd.DataFrame(X_scaled, columns=features)
    
    pred_yield = model.predict(X_scaled)[0]
    
    print("*" * 50)
    print(f"  PREDICTED CROP YIELD FOR {current_year}")
    print(f"  Region: {lat} N, {lon} E")
    print(f"  Yield:  {pred_yield:.2f} tonnes/ha")
    print("*" * 50)

if __name__ == "__main__":
    predict_current_season()
