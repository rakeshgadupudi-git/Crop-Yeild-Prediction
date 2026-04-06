import requests
import pandas as pd
from datetime import datetime
import os

# NASA POWER API Endpoint
# Documentation: https://power.larc.nasa.gov/docs/services/api/
BASE_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"

# API Parameter mappings to model features
PARAM_MAP = {
    'PRECTOTCORR': 'Precipitation',
    'T2M_MAX': 'maxtemp',
    'T2M_MIN': 'mintemp',
    'T2M': 'avgtemp',
    'RH2M': 'relativehumidity',
    'PS': 'surfacepressure',
    'T2MDEW': 'dewpoints',
    'WS2M_MIN': 'minwindspeed',
    'WS2M_MAX': 'maxwindspeed',
    'QV2M': 'specifichumidity',
    'CLOUD_AMT': 'cloudcoverage'
}

def fetch_weather_data(lat, lon, start_year, end_year):
    """
    Fetches monthly weather parameters from NASA POWER API for a given location and year range.
    """
    print(f"Fetching weather data for Lat: {lat}, Lon: {lon} | Period: {start_year}-{end_year}")
    
    parameters = ",".join(PARAM_MAP.keys())
    
    # NASA POWER expects YYYY format for monthly endpoint, but the endpoint requires start and end
    url = f"{BASE_URL}?parameters={parameters}&community=AG&longitude={lon}&latitude={lat}&start={start_year}&end={end_year}&format=JSON"
    
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from NASA POWER API. Status code: {response.status_code}")
    
    data = response.json()
    
    # Extract the monthly data
    # format: { 'properties': { 'parameter': { 'YYYYMM': value, ... } } }
    monthly_data = data['properties']['parameter']
    
    # Construct a DataFrame
    records = []
    
    # get all unique YYYYMM keys from the first parameter
    first_param = list(PARAM_MAP.keys())[0]
    time_keys = monthly_data[first_param].keys()
    
    for tk in time_keys:
        # Ignore Annual ('13') metrics returned by NASA POWER
        if tk.endswith('13'):
            continue
            
        record = {
            'Year': int(tk[:4]),
            'Month': int(tk[4:6])
        }
        
        valid_record = True
        for api_param, model_col in PARAM_MAP.items():
            val = monthly_data[api_param].get(tk, None)
            if val is None or val == -999.0: # NASA POWER uses -999 for missing
                valid_record = False
                break
            record[model_col] = val
            
        if valid_record:
            records.append(record)
            
    df = pd.DataFrame(records)
    
    # Group by Year and take the annual mean (since the model predicts yearly yield)
    # Some features might need sum (e.g. Precipitation), but we will provide both or aggregate based on the research standard.
    # The existing pipeline seems to merge monthly averages or uses max/min for specific months. For simplicity, we create annual aggregates here:
    
    if len(df) == 0:
        print("Warning: No valid data found for the specified period.")
        return df

    annual_df = df.groupby('Year').agg({
        'Precipitation': 'sum',
        'maxtemp': 'max',
        'mintemp': 'min',
        'avgtemp': 'mean',
        'relativehumidity': 'mean',
        'surfacepressure': 'mean',
        'dewpoints': 'mean',
        'minwindspeed': 'min',
        'maxwindspeed': 'max',
        'specifichumidity': 'mean',
        'cloudcoverage': 'mean'
    }).reset_index()
    
    return annual_df

if __name__ == "__main__":
    # Test for Satara District (Vidani village)
    LAT = 17.9824
    LON = 74.5113
    
    current_year = datetime.now().year
    
    # NASA POWER data might not be available for the current year yet (422 error), use current_year - 1
    df_weather = fetch_weather_data(LAT, LON, 2015, current_year - 1)
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Phase3', 'csv')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'realtime_weather_satara.csv')
    df_weather.to_csv(output_path, index=False)
    
    print(f"Successfully fetched and saved {len(df_weather)} years of weather data to {output_path}")
    print(df_weather.tail())
