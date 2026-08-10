"""
Demand Prediction Regression Model Training Script for ECOBITE
Trains Random Forest Regressor to forecast expected product sales demand before expiry.
Evaluates models using MAE, RMSE, and R-squared metrics.
"""

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_preprocessing import prepare_features_and_targets

def train_demand_regression_model(data_path: str = None, models_dir: str = None):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if data_path is None:
        data_path = os.path.join(project_root, "data", "raw", "supermarket_inventory.csv")
    if models_dir is None:
        models_dir = os.path.join(project_root, "models")

    os.makedirs(models_dir, exist_ok=True)

    print(f"Loading data for demand model from {data_path}...")
    df = pd.read_csv(data_path)

    # Calculate demand target before expiry: Daily_Sales velocity * Days_to_Expiry
    df["Expected_Demand_Target"] = np.maximum(
        1.0,
        df["Daily_Sales"].astype(float) * df["Days_to_Expiry"].astype(float)
    )

    X, _ = prepare_features_and_targets(df, target_col="Expected_Demand_Target")
    y = df["Expected_Demand_Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load preprocessor fit on training data
    preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
        X_train_trans = preprocessor.transform(X_train)
        X_test_trans = preprocessor.transform(X_test)
    else:
        from src.data_preprocessing import build_preprocessor_pipeline
        preprocessor = build_preprocessor_pipeline()
        X_train_trans = preprocessor.fit_transform(X_train)
        X_test_trans = preprocessor.transform(X_test)

    rf_reg = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        random_state=42
    )
    rf_reg.fit(X_train_trans, y_train)

    y_pred = rf_reg.predict(X_test_trans)
    y_pred = np.maximum(0, y_pred) # Non-negative constraint

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    reg_metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    print("\n--- Demand Prediction Model (Random Forest Regressor) Results ---")
    print(f"MAE  : {mae:.4f} units")
    print(f"RMSE : {rmse:.4f} units")
    print(f"R^2  : {r2:.4f}")

    demand_model_path = os.path.join(models_dir, "demand_model.pkl")
    meta_path = os.path.join(models_dir, "demand_model_meta.pkl")

    joblib.dump(rf_reg, demand_model_path)
    joblib.dump(reg_metrics, meta_path)

    print(f"\nDemand model successfully saved to: {demand_model_path}")
    return reg_metrics

if __name__ == "__main__":
    train_demand_regression_model()
