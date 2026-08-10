"""
Verification Test Script for ECOBITE End-to-End Pipeline
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.prediction import EcoBitePredictor

def test_sunscreen_scenario():
    predictor = EcoBitePredictor()
    assert predictor.is_loaded, "Predictor models failed to load!"

    sunscreen_input = {
        "Product_ID": "PRD-9999",
        "Product_Name": "Sunscreen Lotion SPF 50",
        "Category": "Personal Care",
        "Current_Stock": 140,
        "Historical_Sales": 100,
        "Daily_Sales": 7.0,
        "Selling_Price": 500.0,
        "Previous_Discount": 10.0,
        "Sales_During_Discount": 8.5,
        "Season": "Summer",
        "Month": 6,
        "Days_to_Expiry": 15,
        "Supplier_Lead_Time": 5,
        "Promotion": 1
    }

    result = predictor.predict_single(sunscreen_input)

    print("\n--- Sunscreen Scenario End-to-End Test Output ---")
    print(f"Product           : {result['Product_Name']}")
    print(f"Current Stock     : {result['Current_Stock']}")
    print(f"Predicted Demand  : {result['Predicted_Demand']}")
    print(f"Excess Stock      : {result['Excess_Inventory']}")
    print(f"Days to Expiry    : {result['Days_to_Expiry']}")
    print(f"Waste Probability : {result['Waste_Probability']*100:.1f}%")
    print(f"Risk Level        : {result['Risk_Level']}")
    print(f"Rec. Discount     : {result['Recommended_Discount']}%")
    print(f"Rec. Selling Price: Rs.{result['Recommended_Price']}")
    print(f"Rec. Action       : {result['Recommended_Action']}")
    print(f"Reorder Decision  : {result['Reorder_Status']}")
    print(f"AI Explanation    :\n{result['Explanation']}")

    assert result['Waste_Probability'] > 0.5, "Waste probability should be high!"
    assert result['Recommended_Discount'] >= 10.0, "Discount should be recommended!"
    print("\n[SUCCESS] Sunscreen scenario assertion tests passed!")

if __name__ == "__main__":
    test_sunscreen_scenario()
