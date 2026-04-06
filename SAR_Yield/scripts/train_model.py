import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

def create_features(df):
    """
    Adds advanced time-series features to the dataset.
    """
    df = df.sort_values('Year').reset_index(drop=True)
    
    # Add a lag feature for Yield (Yield_t-1)
    df['Yield_lag1'] = df['Yield'].shift(1)
    
    # Backward fill the first year's lag using the current year's yield
    df['Yield_lag1'].fillna(df['Yield'], inplace=True)
    
    # 3-year moving average of Precipitation and Temperature
    df['Precip_rolling3'] = df['Precipitation'].rolling(window=3, min_periods=1).mean()
    df['Temp_rolling3'] = df['avgtemp'].rolling(window=3, min_periods=1).mean()
    
    return df

def train_and_evaluate():
    """
    Trains the XGBoost/GradientBoosting model using TimeSeriesSplit and saves the best model.
    """
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'csv', 'phase3_input_yield_sar.csv')
    dataset = pd.read_csv(csv_path)
    
    # 1. Feature Engineering
    dataset = create_features(dataset)
    
    features = [
        'Precipitation', 'maxtemp', 'mintemp', 'avgtemp',
        'specifichumidity', 'relativehumidity', 'surfacepressure',
        'dewpoints', 'minwindspeed', 'maxwindspeed', 'cloudcoverage',
        'Nitrogen', 'Phosphorus', 'Pottasium', 'pH',
        'VH_mean', 'VV_mean', 'RVI', 
        'Yield_lag1', 'Precip_rolling3', 'Temp_rolling3'
    ]
    
    X = dataset[features]
    y = dataset['Yield']
    
    # 2. Imputation & Scaling
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    X_scaled = pd.DataFrame(X_scaled, columns=features)
    
    # 3. Model & Hyperparameter Tuning using TimeSeriesSplit
    # HistGradientBoosting handles NaNs natively and is much faster
    gbm = HistGradientBoostingRegressor(random_state=42)
    
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [2, 3, 5],
        'max_iter': [50, 100, 200]
    }
    
    # Note: Dataset is very small, so we use max possible splits
    tscv = TimeSeriesSplit(n_splits=3)
    
    grid_search = GridSearchCV(
        estimator=gbm, 
        param_grid=param_grid, 
        cv=tscv, 
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    print("Training Gradient Boosting model with TimeSeriesSplit and GridSearch...")
    grid_search.fit(X_scaled, y)
    
    best_model = grid_search.best_estimator_
    print(f"Best hyperparameters found: {grid_search.best_params_}")
    
    # Evaluate the best model on the training set (since data is small)
    predictions = best_model.predict(X_scaled)
    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)
    
    print("\nModel Evaluation Metrics (Full Dataset):")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")
    
    # 4. Save the components for Real-time Inference
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'best_yield_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    imputer_path = os.path.join(model_dir, 'imputer.pkl')
    
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(imputer, imputer_path)
    
    print(f"\nModel successfully saved to {model_path}")
    print(f"Scaler successfully saved to {scaler_path}")

if __name__ == "__main__":
    train_and_evaluate()
