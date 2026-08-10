"""
Feature Engineering Module for ECOBITE
Extracts key domain metrics including sales velocity, stock-to-demand ratio, sell-through, and expiry urgency.
"""

import pandas as pd
import numpy as np

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific engineered features for inventory waste and demand modeling.
    Returns a copy of the dataframe with added feature columns.
    """
    data = df.copy()

    # 1. Sales Velocity (units sold per day)
    if "Daily_Sales" in data.columns:
        data["Sales_Velocity"] = data["Daily_Sales"].astype(float)
    elif "Historical_Sales" in data.columns:
        data["Sales_Velocity"] = (data["Historical_Sales"] / 30.0).astype(float)
    else:
        data["Sales_Velocity"] = 1.0

    # 2. Expiry Urgency Index (Higher value = higher urgency/closer to expiry)
    data["Expiry_Urgency"] = 1.0 / (data["Days_to_Expiry"].astype(float) + 1.0)

    # 3. Stock to Historical Demand Ratio
    hist_sales = np.maximum(data["Historical_Sales"].astype(float), 1.0)
    data["Stock_to_Demand_Ratio"] = data["Current_Stock"].astype(float) / hist_sales

    # 4. Estimated Sell-Through Ratio before expiry (Naive projected sales / stock)
    projected_naive_sales = data["Sales_Velocity"] * data["Days_to_Expiry"].astype(float)
    current_stock_safe = np.maximum(data["Current_Stock"].astype(float), 1.0)
    data["Estimated_Sell_Through"] = np.minimum(1.0, projected_naive_sales / current_stock_safe)

    # 5. Stock Coverage Days (How many days of stock remaining at current velocity)
    sales_vel_safe = np.maximum(data["Sales_Velocity"], 0.1)
    data["Stock_Coverage_Days"] = data["Current_Stock"].astype(float) / sales_vel_safe

    # 6. Expiry Deficit Days (Coverage days minus days to expiry)
    data["Expiry_Deficit_Days"] = data["Stock_Coverage_Days"] - data["Days_to_Expiry"].astype(float)

    # 7. Recent Sales Trend Index (Comparing daily rate to historical monthly average)
    monthly_avg_daily_sales = hist_sales / 30.0
    data["Recent_Sales_Trend"] = data["Sales_Velocity"] / np.maximum(monthly_avg_daily_sales, 0.1)

    # 8. Historical Discount Responsiveness (Elasticity ratio)
    prev_disc = data["Previous_Discount"].astype(float)
    sales_during_disc = data["Sales_During_Discount"].astype(float)
    data["Discount_Responsiveness"] = np.where(
        prev_disc > 0,
        (sales_during_disc - data["Sales_Velocity"]) / np.maximum(prev_disc, 1.0),
        0.0
    )

    # 9. Seasonal Index (Encoding season affinity)
    season_weights = {"Summer": 1.2, "Winter": 1.1, "Monsoon": 0.9, "Spring": 1.0}
    data["Season_Weight"] = data["Season"].map(season_weights).fillna(1.0)

    return data

def get_feature_columns():
    """
    Returns lists of numerical and categorical feature names used for model input.
    """
    num_cols = [
        "Current_Stock", "Historical_Sales", "Daily_Sales", "Selling_Price",
        "Previous_Discount", "Sales_During_Discount", "Month", "Days_to_Expiry",
        "Supplier_Lead_Time", "Promotion", "Sales_Velocity", "Expiry_Urgency",
        "Stock_to_Demand_Ratio", "Estimated_Sell_Through", "Stock_Coverage_Days",
        "Expiry_Deficit_Days", "Recent_Sales_Trend", "Discount_Responsiveness",
        "Season_Weight"
    ]
    cat_cols = ["Category", "Season"]
    return num_cols, cat_cols
