"""
Unified Prediction & Decision-Support Pipeline for ECOBITE
Loads saved scikit-learn models and runs end-to-end inference on single product dicts or batch CSV dataframes.
"""

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import joblib
import pandas as pd
import numpy as np

from src.feature_engineering import add_engineered_features, get_feature_columns
from src.discount_engine import calculate_dynamic_discount
from src.recommendation_engine import generate_inventory_recommendation

class EcoBitePredictor:
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(project_root, "models")

        self.models_dir = models_dir
        self.waste_model = None
        self.demand_model = None
        self.preprocessor = None
        self.is_loaded = False
        
        self.load_models()

    def load_models(self):
        waste_path = os.path.join(self.models_dir, "waste_model.pkl")
        demand_path = os.path.join(self.models_dir, "demand_model.pkl")
        prep_path = os.path.join(self.models_dir, "preprocessor.pkl")

        if os.path.exists(waste_path) and os.path.exists(demand_path) and os.path.exists(prep_path):
            self.waste_model = joblib.load(waste_path)
            self.demand_model = joblib.load(demand_path)
            self.preprocessor = joblib.load(prep_path)
            self.is_loaded = True
            return True
        else:
            self.is_loaded = False
            return False

    def predict_single(self, product_dict: dict) -> dict:
        """
        Runs prediction on a single product input dictionary.
        """
        df_single = pd.DataFrame([product_dict])
        results_df = self.predict_dataframe(df_single)
        return results_df.iloc[0].to_dict()

    def predict_dataframe(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Runs batch prediction, demand estimation, excess inventory calculation,
        discount recommendation, and action generation on a pandas DataFrame.
        """
        df_clean = df_input.copy()
        defaults = {
            "Unit": "units",
            "Previous_Discount": 0.0,
            "Supplier_Lead_Time": 5,
            "Promotion": 0,
            "Month": 6,
            "Season": "All-Season",
            "Historical_Sales": 100.0,
            "Daily_Sales": 3.3
        }
        for col, val in defaults.items():
            if col not in df_clean.columns:
                df_clean[col] = val
        if "Sales_During_Discount" not in df_clean.columns:
            df_clean["Sales_During_Discount"] = df_clean["Daily_Sales"]
            
        df_feat = add_engineered_features(df_clean)
        num_cols, cat_cols = get_feature_columns()
        X = df_feat[num_cols + cat_cols]

        X_trans = self.preprocessor.transform(X)

        # 1. Waste Classification
        waste_probs = self.waste_model.predict_proba(X_trans)[:, 1]
        waste_preds = self.waste_model.predict(X_trans)

        # 2. Demand Regression
        demand_preds = self.demand_model.predict(X_trans)
        demand_preds = np.maximum(0, demand_preds) # Non-negative demand constraint

        results = df_input.copy()
        results["Predicted_Demand"] = np.round(demand_preds, 1)
        results["Waste_Probability"] = np.round(waste_probs, 4)
        results["Waste_Prediction"] = waste_preds

        # Calculate Excess, Discounts, and Actions per row
        excess_list = []
        risk_levels = []
        disc_pcts = []
        disc_prices = []
        actions = []
        reorder_statuses = []
        explanations = []
        waste_saved_values = []

        for idx, row in results.iterrows():
            stock = float(row["Current_Stock"])
            pred_demand = float(row["Predicted_Demand"])
            days_exp = float(row["Days_to_Expiry"])
            daily_sales = float(row.get("Daily_Sales", 1.0))
            price = float(row.get("Selling_Price", 100.0))
            prev_disc = float(row.get("Previous_Discount", 0.0))
            sales_disc = float(row.get("Sales_During_Discount", daily_sales))
            p_name = str(row.get("Product_Name", f"Product {row.get('Product_ID', idx)}"))
            season = str(row.get("Season", "All-Season"))
            w_prob = float(row["Waste_Probability"])

            # Discount Engine
            disc_res = calculate_dynamic_discount(
                waste_probability=w_prob,
                current_stock=stock,
                predicted_demand=pred_demand,
                days_to_expiry=days_exp,
                daily_sales=daily_sales,
                selling_price=price,
                previous_discount=prev_disc,
                sales_during_discount=sales_disc
            )

            # Recommendation Engine
            rec_res = generate_inventory_recommendation(
                product_name=p_name,
                waste_probability=w_prob,
                risk_level=disc_res["risk_level"],
                current_stock=stock,
                predicted_demand=pred_demand,
                excess_stock=disc_res["excess_stock"],
                days_to_expiry=days_exp,
                recommended_discount_pct=disc_res["recommended_discount_pct"],
                season=season
            )

            excess_list.append(disc_res["excess_stock"])
            risk_levels.append(disc_res["risk_level"])
            disc_pcts.append(disc_res["recommended_discount_pct"])
            disc_prices.append(disc_res["discounted_price"])
            actions.append(rec_res["action"])
            reorder_statuses.append(rec_res["reorder_status"])
            explanations.append(rec_res["explanation"])
            waste_saved_values.append(disc_res["waste_saved_value"])

        results["Excess_Inventory"] = excess_list
        results["Risk_Level"] = risk_levels
        results["Recommended_Discount"] = disc_pcts
        results["Recommended_Price"] = disc_prices
        results["Recommended_Action"] = actions
        results["Reorder_Status"] = reorder_statuses
        results["Explanation"] = explanations
        results["Potential_Savings_INR"] = waste_saved_values

        return results
