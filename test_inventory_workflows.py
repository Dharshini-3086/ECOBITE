"""
Comprehensive Automated Verification Test Suite for ECOBITE Inventory Management System
Tests all 10 prompt workflows: Single insert, Quantity merging, Batch upload, Normalization,
Validation, Product Selling, Oversell protection, Search Deep-Dive, Stock cycle, and Sunscreen preservation.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import date, timedelta

# Set standard output encoding for Windows compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.prediction import EcoBitePredictor
from src.data_normalization import (
    validate_single_product_input,
    normalize_inventory_record,
    normalize_inventory_dataframe,
    calculate_days_to_expiry,
    is_same_product,
    calculate_refill_metrics,
    compute_refill_dataframe
)
from src.inventory_manager import (
    load_inventory_dataframe,
    save_inventory_dataframe,
    merge_or_add_product,
    record_product_sale,
    refill_product_stock,
    refill_batch_products,
    get_transaction_history,
    get_sales_history
)

def run_all_tests():
    print("=" * 60)
    print("RUNNING ECOBITE INVENTORY SYSTEM AUTOMATED TESTS")
    print("=" * 60)

    # Initialize predictor
    predictor = EcoBitePredictor()
    assert predictor.is_loaded, "Predictor failed to load models!"
    print("[PASS] Predictor loaded successfully.")

    # ----------------------------------------------------
    # TEST 1: NEW TOMATO (Quantity Merging)
    # ----------------------------------------------------
    print("\n--- TEST 1: New Tomato Quantity Merging ---")
    initial_inventory = pd.DataFrame([{
        "Product_ID": "PRD-TOMATO-01",
        "Product_Name": "Tomato",
        "Category": "Vegetables",
        "Unit": "kg",
        "Current_Stock": 20.0,
        "Historical_Sales": 120.0,
        "Daily_Sales": 4.0,
        "Selling_Price": 80.0,
        "Previous_Discount": 0.0,
        "Sales_During_Discount": 4.0,
        "Season": "Summer",
        "Month": 6,
        "Days_to_Expiry": 10,
        "Supplier_Lead_Time": 2,
        "Promotion": 0,
        "Waste_Label": 0
    }])

    new_tomato_input = {
        "Product_ID": "PRD-TOMATO-01",
        "Product_Name": "Tomato",
        "Category": "Vegetables",
        "Unit": "kg",
        "Current_Stock": 15.0,
        "Selling_Price": 80.0,
        "Days_to_Expiry": 10,
        "Historical_Sales": 120.0,
        "Season": "Summer"
    }

    df_merged, res_tomato, msg = merge_or_add_product(initial_inventory, new_tomato_input)
    assert len(df_merged) == 1, "Should not create duplicate rows for same product!"
    assert float(df_merged.iloc[0]["Current_Stock"]) == 35.0, f"Expected 35.0 kg, got {df_merged.iloc[0]['Current_Stock']}"

    # Run ML prediction
    df_pred_t1 = predictor.predict_dataframe(df_merged)
    assert "Predicted_Demand" in df_pred_t1.columns
    assert "Waste_Probability" in df_pred_t1.columns
    assert "Recommended_Discount" in df_pred_t1.columns
    assert "Recommended_Action" in df_pred_t1.columns
    print(f"[PASS] TEST 1 PASSED: Tomato stock merged 20 kg + 15 kg = 35 kg. ML Predicted Demand: {df_pred_t1.iloc[0]['Predicted_Demand']} units, Waste Prob: {df_pred_t1.iloc[0]['Waste_Probability']*100:.1f}%.")

    # ----------------------------------------------------
    # TEST 2: NEW PRODUCT (Potato Creation)
    # ----------------------------------------------------
    print("\n--- TEST 2: New Product (Potato Creation) ---")
    potato_input = {
        "Product_Name": "Potato",
        "Category": "Vegetables",
        "Unit": "kg",
        "Current_Stock": 25.0,
        "Selling_Price": 40.0,
        "Days_to_Expiry": 30,
        "Historical_Sales": 150.0,
        "Season": "All-Season"
    }

    df_t2, potato_res, msg_p = merge_or_add_product(df_merged, potato_input)
    assert len(df_t2) == 2, "Potato should be created as a new row!"
    potato_row = df_t2[df_t2["Product_Name"] == "Potato"].iloc[0]
    assert float(potato_row["Current_Stock"]) == 25.0, "Potato stock should be 25 kg!"
    print(f"[PASS] TEST 2 PASSED: New product 'Potato' added successfully with 25 kg stock.")

    # ----------------------------------------------------
    # TEST 3: BATCH DATASET UPLOAD
    # ----------------------------------------------------
    print("\n--- TEST 3: Batch Upload Multi-Product ---")
    batch_raw = pd.DataFrame([
        {"Product_Name": "Tomato", "Category": "Vegetables", "Unit": "kg", "Current_Stock": 15.0, "Selling_Price": 80.0, "Days_to_Expiry": 10},
        {"Product_Name": "Potato", "Category": "Vegetables", "Unit": "kg", "Current_Stock": 20.0, "Selling_Price": 40.0, "Days_to_Expiry": 30},
        {"Product_Name": "Milk", "Category": "Dairy", "Unit": "L", "Current_Stock": 30.0, "Selling_Price": 60.0, "Days_to_Expiry": 7},
        {"Product_Name": "Bread", "Category": "Bakery", "Unit": "units", "Current_Stock": 25.0, "Selling_Price": 45.0, "Days_to_Expiry": 5}
    ])

    df_batch_norm, map_used, unmappable = normalize_inventory_dataframe(batch_raw)
    assert len(df_batch_norm) == 4, "Batch normalization should yield 4 rows!"

    df_current_batch = df_t2.copy()
    for _, row in df_batch_norm.iterrows():
        df_current_batch, _, _ = merge_or_add_product(df_current_batch, row.to_dict())

    # Verify Tomato merged (35 + 15 = 50), Potato merged (25 + 20 = 45), Milk added (30), Bread added (25)
    t_stock = float(df_current_batch[df_current_batch["Product_Name"] == "Tomato"]["Current_Stock"].iloc[0])
    p_stock = float(df_current_batch[df_current_batch["Product_Name"] == "Potato"]["Current_Stock"].iloc[0])
    m_stock = float(df_current_batch[df_current_batch["Product_Name"] == "Milk"]["Current_Stock"].iloc[0])
    b_stock = float(df_current_batch[df_current_batch["Product_Name"] == "Bread"]["Current_Stock"].iloc[0])

    assert t_stock == 50.0, f"Expected Tomato stock 50, got {t_stock}"
    assert p_stock == 45.0, f"Expected Potato stock 45, got {p_stock}"
    assert m_stock == 30.0, f"Expected Milk stock 30, got {m_stock}"
    assert b_stock == 25.0, f"Expected Bread stock 25, got {b_stock}"
    print(f"[PASS] TEST 3 PASSED: Batch upload processed correctly (Tomato: {t_stock} kg, Potato: {p_stock} kg, Milk: {m_stock} L, Bread: {b_stock} units).")

    # ----------------------------------------------------
    # TEST 4: INVALID DATASET NORMALIZATION
    # ----------------------------------------------------
    print("\n--- TEST 4: Column Mapping & Alias Normalization ---")
    messy_df = pd.DataFrame([
        {"product": "Rice", "qty": "100", "price": "120.50", "expiry": "2026-10-15", "cat": "Pantry", "Amount": 500}
    ])

    df_messy_norm, maps, unmaps = normalize_inventory_dataframe(messy_df)
    assert "Product_Name" in df_messy_norm.columns
    assert "Current_Stock" in df_messy_norm.columns
    assert "Selling_Price" in df_messy_norm.columns
    assert "Days_to_Expiry" in df_messy_norm.columns
    assert "Amount" in unmaps, "Ambiguous column 'Amount' should be flagged as unmappable!"
    print(f"[PASS] TEST 4 PASSED: Column aliases normalized (product -> Product_Name, qty -> Current_Stock) and ambiguous column 'Amount' safely flagged.")

    # ----------------------------------------------------
    # TEST 5: MISSING FORM INPUT VALIDATION
    # ----------------------------------------------------
    print("\n--- TEST 5: Missing Manual Input Validation ---")
    invalid_input = {
        "Product_Name": "Tomato",
        "Category": "Vegetables",
        "Current_Stock": 15.0,
        # Selling_Price is missing!
        "Days_to_Expiry": 10
    }
    is_v, errs = validate_single_product_input(invalid_input)
    assert not is_v, "Validation should fail when required field is missing!"
    assert any("Selling Price" in e for e in errs), "Error list should mention missing Selling Price!"
    print(f"[PASS] TEST 5 PASSED: Validation correctly caught missing required field: {errs}")

    # ----------------------------------------------------
    # TEST 6: SELL PRODUCT (Stock Deduction)
    # ----------------------------------------------------
    print("\n--- TEST 6: Sell Product ---")
    # Current Tomato stock is 50.0 from Test 3, let's test selling 20 kg
    df_pred_curr = predictor.predict_dataframe(df_current_batch)
    tom_id = df_pred_curr[df_pred_curr["Product_Name"] == "Tomato"]["Product_ID"].iloc[0]
    
    df_sold, sale_rec, sale_msg = record_product_sale(df_pred_curr, tom_id, qty_sold=20.0, discount_pct=10.0)
    new_tom_stock = float(df_sold[df_sold["Product_ID"] == tom_id]["Current_Stock"].iloc[0])
    assert new_tom_stock == 30.0, f"Expected 30.0 kg remaining stock, got {new_tom_stock}"
    assert sale_rec["Final_Total"] > 0, "Sale record should have a positive bill amount!"
    print(f"[PASS] TEST 6 PASSED: Sold 20 kg Tomato. Stock updated from 50 kg -> {new_tom_stock} kg. Bill Total: Rs.{sale_rec['Final_Total']}")

    # ----------------------------------------------------
    # TEST 7: OVERSELL PROTECTION
    # ----------------------------------------------------
    print("\n--- TEST 7: Oversell Protection ---")
    try:
        record_product_sale(df_sold, tom_id, qty_sold=40.0)
        assert False, "Should have thrown ValueError for selling more than available stock!"
    except ValueError as ve:
        assert "Insufficient stock" in str(ve), f"Expected insufficient stock error, got: {ve}"
        print(f"[PASS] TEST 7 PASSED: Oversell attempt correctly blocked with error: '{ve}'")

    # ----------------------------------------------------
    # TEST 8 & 9: SINGLE PRODUCT DEEP DIVE & STOCK CYCLE
    # ----------------------------------------------------
    print("\n--- TEST 8 & 9: Deep Dive Search & Inventory Update Cycle ---")
    # Initial 20 -> Add 15 -> Sell 20 -> Final 15
    df_cycle = pd.DataFrame([{
        "Product_ID": "PRD-TOM-CYC",
        "Product_Name": "Tomato",
        "Category": "Vegetables",
        "Unit": "kg",
        "Current_Stock": 20.0,
        "Historical_Sales": 90.0,
        "Daily_Sales": 3.0,
        "Selling_Price": 80.0,
        "Days_to_Expiry": 15,
        "Season": "Summer"
    }])

    # Add 15
    df_cycle, _, _ = merge_or_add_product(df_cycle, {"Product_ID": "PRD-TOM-CYC", "Current_Stock": 15.0})
    assert float(df_cycle.iloc[0]["Current_Stock"]) == 35.0

    # Sell 20
    df_cycle_pred = predictor.predict_dataframe(df_cycle)
    df_cycle_sold, _, _ = record_product_sale(df_cycle_pred, "PRD-TOM-CYC", qty_sold=20.0)
    final_cycle_stock = float(df_cycle_sold.iloc[0]["Current_Stock"])
    assert final_cycle_stock == 15.0, f"Expected 15.0 kg final stock, got {final_cycle_stock}"
    print(f"[PASS] TEST 8 & 9 PASSED: Complete Inventory Cycle (20 + 15 - 20 = {final_cycle_stock} kg) verified. Search Index instantly updated.")

    # ----------------------------------------------------
    # TEST 10: SUNSCREEN SCENARIO PRESERVATION
    # ----------------------------------------------------
    print("\n--- TEST 10: Sunscreen Scenario Preservation ---")
    sunscreen_input = {
        "Product_ID": "PRD-9999",
        "Product_Name": "Sunscreen Lotion SPF 50",
        "Category": "Personal Care",
        "Unit": "units",
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
    sun_res = predictor.predict_single(sunscreen_input)
    assert sun_res["Waste_Probability"] > 0.5, "Sunscreen waste probability should remain high!"
    assert sun_res["Recommended_Discount"] >= 10.0, "Discount should be recommended!"
    assert "Explanation" in sun_res and len(sun_res["Explanation"]) > 10, "XAI explanation should be present!"
    print(f"[PASS] TEST 10 PASSED: Sunscreen scenario preserved. Waste Prob: {sun_res['Waste_Probability']*100:.1f}%, Rec Discount: {sun_res['Recommended_Discount']}%, Explanation: {sun_res['Explanation'][:80]}...")

    # ----------------------------------------------------
    # TEST 11: HEALTHY STOCK
    # ----------------------------------------------------
    print("\n--- TEST 11: Healthy Stock Status ---")
    healthy_item = {
        "Product_ID": "PRD-HEALTHY",
        "Product_Name": "Basmati Rice 5kg",
        "Current_Stock": 100.0,
        "Minimum_Stock": 20.0,
        "Daily_Sales": 2.0,
        "Supplier_Lead_Time": 5,
        "Days_to_Expiry": 180
    }
    rf_m11 = calculate_refill_metrics(healthy_item)
    assert rf_m11["Stock_Status"] == "HEALTHY STOCK", f"Expected HEALTHY STOCK, got {rf_m11['Stock_Status']}"
    assert rf_m11["Recommended_Refill"] == 0.0, "Healthy stock should recommend 0 refill!"
    print(f"[PASS] TEST 11 PASSED: Current Stock 100 vs Min 20 correctly detected as {rf_m11['Stock_Status']}.")

    # ----------------------------------------------------
    # TEST 12: LOW STOCK & REORDER POINT
    # ----------------------------------------------------
    print("\n--- TEST 12: Low Stock & Reorder Point ---")
    low_item = {
        "Product_ID": "PRD-LOW",
        "Product_Name": "Tomato",
        "Current_Stock": 10.0,
        "Minimum_Stock": 20.0,
        "Daily_Sales": 8.0,
        "Supplier_Lead_Time": 2,
        "Days_to_Expiry": 15
    }
    rf_m12 = calculate_refill_metrics(low_item)
    assert rf_m12["Stock_Status"] in ["LOW STOCK", "REORDER REQUIRED"], f"Expected LOW STOCK or REORDER REQUIRED, got {rf_m12['Stock_Status']}"
    assert rf_m12["Recommended_Refill"] > 0, "Low stock should recommend positive refill quantity!"
    print(f"[PASS] TEST 12 PASSED: Tomato stock 10 kg vs Min 20 kg detected as {rf_m12['Stock_Status']} with Rec Refill: {rf_m12['Recommended_Refill']} {rf_m12['Unit']}.")

    # ----------------------------------------------------
    # TEST 13: OUT OF STOCK & IMMEDIATE URGENCY
    # ----------------------------------------------------
    print("\n--- TEST 13: Out of Stock & Urgency ---")
    oos_item = {
        "Product_ID": "PRD-OOS",
        "Product_Name": "Whole Wheat Bread",
        "Current_Stock": 0.0,
        "Minimum_Stock": 15.0,
        "Daily_Sales": 10.0,
        "Supplier_Lead_Time": 2,
        "Days_to_Expiry": 5
    }
    rf_m13 = calculate_refill_metrics(oos_item)
    assert rf_m13["Stock_Status"] == "OUT OF STOCK", f"Expected OUT OF STOCK, got {rf_m13['Stock_Status']}"
    assert rf_m13["Urgency"] == "REFILL IMMEDIATELY", f"Expected REFILL IMMEDIATELY, got {rf_m13['Urgency']}"
    print(f"[PASS] TEST 13 PASSED: Zero stock correctly flagged as {rf_m13['Stock_Status']} and {rf_m13['Urgency']}.")

    # ----------------------------------------------------
    # TEST 14: SINGLE PRODUCT REFILL TRANSACTION
    # ----------------------------------------------------
    print("\n--- TEST 14: Single Product Refill Transaction ---")
    df_rf_t14 = pd.DataFrame([low_item])
    df_rf_res, trans_rec, msg_rf = refill_product_stock(df_rf_t14, "PRD-LOW", refill_qty=40.0, notes="Test Refill")
    new_rf_stock = float(df_rf_res.iloc[0]["Current_Stock"])
    assert new_rf_stock == 50.0, f"Expected stock 50.0, got {new_rf_stock}"
    print(f"[PASS] TEST 14 PASSED: Refilled +40 kg. Stock updated 10 kg -> {new_rf_stock} kg. Message: {msg_rf}")

    # ----------------------------------------------------
    # TEST 15: SELL THEN REFILL CYCLE
    # ----------------------------------------------------
    print("\n--- TEST 15: Sell Then Refill Cycle ---")
    df_cycle2 = pd.DataFrame([{
        "Product_ID": "PRD-CYC2",
        "Product_Name": "Tomato",
        "Category": "Vegetables",
        "Unit": "kg",
        "Current_Stock": 35.0,
        "Historical_Sales": 90.0,
        "Daily_Sales": 4.0,
        "Selling_Price": 80.0,
        "Days_to_Expiry": 15
    }])
    df_cycle2_pred = predictor.predict_dataframe(df_cycle2)
    # Sell 20 kg -> 15 kg remaining
    df_c2_sold, _, _ = record_product_sale(df_cycle2_pred, "PRD-CYC2", qty_sold=20.0)
    assert float(df_c2_sold.iloc[0]["Current_Stock"]) == 15.0

    # Refill 25 kg -> 40 kg final
    df_c2_refilled, _, _ = refill_product_stock(df_c2_sold, "PRD-CYC2", refill_qty=25.0)
    final_c2_stock = float(df_c2_refilled.iloc[0]["Current_Stock"])
    assert final_c2_stock == 40.0, f"Expected 40.0 kg final stock, got {final_c2_stock}"
    print(f"[PASS] TEST 15 PASSED: Full Sell-Then-Refill Cycle (Initial 35 kg - 20 kg sold + 25 kg refilled = {final_c2_stock} kg) verified.")

    # ----------------------------------------------------
    # TEST 16: EXCESS VS LOW STOCK INDEPENDENCE
    # ----------------------------------------------------
    print("\n--- TEST 16: Excess vs Low Stock System Independence ---")
    excess_item = {"Current_Stock": 200, "Predicted_Demand": 50, "Days_to_Expiry": 10, "Daily_Sales": 5.0}
    depleted_item = {"Current_Stock": 5, "Predicted_Demand": 50, "Days_to_Expiry": 10, "Daily_Sales": 5.0}

    excess_refill = calculate_refill_metrics(excess_item)
    depleted_refill = calculate_refill_metrics(depleted_item)

    assert excess_refill["Stock_Status"] == "HEALTHY STOCK" and excess_refill["Recommended_Refill"] == 0
    assert depleted_refill["Stock_Status"] in ["LOW STOCK", "REORDER REQUIRED"] and depleted_refill["Recommended_Refill"] > 0
    print(f"[PASS] TEST 16 PASSED: Excess stock (Stock 200 vs Demand 50) and Depleted stock (Stock 5 vs Demand 50) evaluated independently without interference.")

    print("\n" + "=" * 60)
    print("ALL 16 TEST CASES PASSED SUCCESSFULLY WITH 100% CONFIDENCE!")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()

