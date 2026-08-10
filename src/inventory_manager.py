"""
Inventory & Transaction Persistence Manager for ECOBITE
Manages CRUD operations, product stock merging, sales transactions,
and audit logs with CSV persistence.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from src.data_normalization import normalize_inventory_record, is_same_product

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
INVENTORY_CSV = os.path.join(DATA_DIR, "supermarket_inventory.csv")
TRANSACTIONS_CSV = os.path.join(DATA_DIR, "inventory_transactions.csv")
SALES_CSV = os.path.join(DATA_DIR, "sales_history.csv")

def ensure_data_files_exist():
    """Ensures data directory and necessary log files exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Transactions log schema
    if not os.path.exists(TRANSACTIONS_CSV):
        df_trans = pd.DataFrame(columns=[
            "Transaction_ID", "Timestamp", "Transaction_Type", "Product_ID",
            "Product_Name", "Category", "Quantity_Change", "Unit",
            "Previous_Stock", "New_Stock", "Notes"
        ])
        df_trans.to_csv(TRANSACTIONS_CSV, index=False)

    # Sales history schema
    if not os.path.exists(SALES_CSV):
        df_sales = pd.DataFrame(columns=[
            "Sale_ID", "Timestamp", "Product_ID", "Product_Name",
            "Quantity_Sold", "Unit", "Selling_Price", "Discount_Pct",
            "Discount_Amount", "Final_Total"
        ])
        df_sales.to_csv(SALES_CSV, index=False)

def load_inventory_dataframe() -> pd.DataFrame:
    """Loads inventory DataFrame from disk."""
    ensure_data_files_exist()
    if os.path.exists(INVENTORY_CSV):
        df = pd.read_csv(INVENTORY_CSV)
        if "Unit" not in df.columns:
            df["Unit"] = "units"
        return df
    else:
        from src.data.raw.generate_synthetic_data import main as gen_data
        gen_data()
        return pd.read_csv(INVENTORY_CSV)

def save_inventory_dataframe(df: pd.DataFrame):
    """Saves updated inventory DataFrame to disk."""
    ensure_data_files_exist()
    df.to_csv(INVENTORY_CSV, index=False)

def log_transaction(trans_type: str, product_id: str, product_name: str, category: str,
                    qty_change: float, unit: str, prev_stock: float, new_stock: float, notes: str):
    """Appends an auditable record to inventory_transactions.csv."""
    ensure_data_files_exist()
    trans_id = f"TXN-{int(datetime.now().timestamp() * 1000)}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_row = {
        "Transaction_ID": trans_id,
        "Timestamp": timestamp,
        "Transaction_Type": trans_type,
        "Product_ID": product_id,
        "Product_Name": product_name,
        "Category": category,
        "Quantity_Change": round(qty_change, 2),
        "Unit": unit,
        "Previous_Stock": round(prev_stock, 2),
        "New_Stock": round(new_stock, 2),
        "Notes": notes
    }

    df_trans = pd.read_csv(TRANSACTIONS_CSV)
    df_trans = pd.concat([pd.DataFrame([new_row]), df_trans], ignore_index=True)
    df_trans.to_csv(TRANSACTIONS_CSV, index=False)

def get_transaction_history() -> pd.DataFrame:
    """Returns inventory transaction audit log."""
    ensure_data_files_exist()
    return pd.read_csv(TRANSACTIONS_CSV)

def get_sales_history() -> pd.DataFrame:
    """Returns sales history log."""
    ensure_data_files_exist()
    return pd.read_csv(SALES_CSV)

def merge_or_add_product(df_current: pd.DataFrame, new_product_dict: dict) -> tuple[pd.DataFrame, dict, str]:
    """
    Merges quantity if product already exists in inventory, or appends a new product row.
    Returns (updated_df, resulting_product_dict, action_taken_message).
    """
    df = df_current.copy()
    norm_p = normalize_inventory_record(new_product_dict)

    # Check for existing match
    matched_idx = None
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        if is_same_product(row_dict, norm_p):
            matched_idx = idx
            break

    if matched_idx is not None:
        # Existing product found -> Check unit compatibility
        existing_unit = str(df.loc[matched_idx, "Unit"]).strip().lower() if "Unit" in df.columns else "units"
        new_unit = str(norm_p.get("Unit", "units")).strip().lower()

        if existing_unit != new_unit and existing_unit != "units" and new_unit != "units":
            raise ValueError(f"Cannot merge inventory because units are different ({existing_unit} vs {new_unit}).")

        prev_stock = float(df.loc[matched_idx, "Current_Stock"])
        added_stock = float(norm_p["Current_Stock"])
        new_stock = prev_stock + added_stock

        df.loc[matched_idx, "Current_Stock"] = new_stock

        # Update secondary fields if provided
        for key in ["Selling_Price", "Days_to_Expiry", "Historical_Sales", "Daily_Sales", "Previous_Discount", "Season"]:
            if key in new_product_dict and new_product_dict[key] is not None:
                df.loc[matched_idx, key] = norm_p[key]

        prod_id = str(df.loc[matched_idx, "Product_ID"])
        prod_name = str(df.loc[matched_idx, "Product_Name"])
        category = str(df.loc[matched_idx, "Category"])

        log_transaction(
            trans_type="MERGE_ADD",
            product_id=prod_id,
            product_name=prod_name,
            category=category,
            qty_change=added_stock,
            unit=new_unit,
            prev_stock=prev_stock,
            new_stock=new_stock,
            notes=f"Merged +{added_stock} {new_unit} into existing stock ({prev_stock} -> {new_stock})"
        )

        msg = f"Merged {added_stock} {new_unit} into existing product '{prod_name}'. Total stock is now {new_stock} {new_unit}."
        res_dict = df.loc[matched_idx].to_dict()
        return df, res_dict, msg

    else:
        # Create new product row
        new_stock = float(norm_p["Current_Stock"])
        df = pd.concat([df, pd.DataFrame([norm_p])], ignore_index=True)

        log_transaction(
            trans_type="NEW_PRODUCT",
            product_id=norm_p["Product_ID"],
            product_name=norm_p["Product_Name"],
            category=norm_p["Category"],
            qty_change=new_stock,
            unit=norm_p["Unit"],
            prev_stock=0.0,
            new_stock=new_stock,
            notes=f"Created new product '{norm_p['Product_Name']}' with initial stock {new_stock} {norm_p['Unit']}."
        )

        msg = f"Added new product '{norm_p['Product_Name']}' to inventory with stock {new_stock} {norm_p['Unit']}."
        return df, norm_p, msg

def record_product_sale(df_current: pd.DataFrame, product_id: str, qty_sold: float, discount_pct: float = None) -> tuple[pd.DataFrame, dict, str]:
    """
    Deducts sold quantity from stock after validating availability.
    Records transaction and sale billing history.
    Returns (updated_df, bill_details_dict, success_message).
    """
    df = df_current.copy()
    matched = df[df["Product_ID"] == product_id]

    if matched.empty:
        raise ValueError(f"Product ID '{product_id}' not found in active inventory.")

    idx = matched.index[0]
    p_name = str(df.loc[idx, "Product_Name"])
    category = str(df.loc[idx, "Category"])
    unit = str(df.loc[idx, "Unit"]) if "Unit" in df.columns else "units"
    current_stock = float(df.loc[idx, "Current_Stock"])
    unit_price = float(df.loc[idx, "Selling_Price"])

    if qty_sold <= 0:
        raise ValueError("Quantity sold must be greater than zero.")

    if qty_sold > current_stock:
        raise ValueError(f"Insufficient stock. Available quantity: {current_stock} {unit}. You cannot sell {qty_sold} {unit}.")

    new_stock = current_stock - qty_sold
    df.loc[idx, "Current_Stock"] = new_stock

    # Discount calculation
    rec_disc = float(df.loc[idx, "Recommended_Discount"]) if "Recommended_Discount" in df.columns else 0.0
    effective_disc_pct = discount_pct if discount_pct is not None else rec_disc

    subtotal = qty_sold * unit_price
    disc_amount = subtotal * (effective_disc_pct / 100.0)
    final_total = subtotal - disc_amount

    # Log sale transaction
    sale_id = f"BILL-{int(datetime.now().timestamp() * 1000)}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sale_record = {
        "Sale_ID": sale_id,
        "Timestamp": timestamp,
        "Product_ID": product_id,
        "Product_Name": p_name,
        "Quantity_Sold": round(qty_sold, 2),
        "Unit": unit,
        "Selling_Price": round(unit_price, 2),
        "Discount_Pct": round(effective_disc_pct, 1),
        "Discount_Amount": round(disc_amount, 2),
        "Final_Total": round(final_total, 2)
    }

    ensure_data_files_exist()
    df_sales = pd.read_csv(SALES_CSV)
    df_sales = pd.concat([pd.DataFrame([sale_record]), df_sales], ignore_index=True)
    df_sales.to_csv(SALES_CSV, index=False)

    log_transaction(
        trans_type="SALE",
        product_id=product_id,
        product_name=p_name,
        category=category,
        qty_change=-qty_sold,
        unit=unit,
        prev_stock=current_stock,
        new_stock=new_stock,
        notes=f"Sold {qty_sold} {unit} under Bill {sale_id}. Remaining stock: {new_stock} {unit}."
    )

    msg = f"Completed sale of {qty_sold} {unit} for '{p_name}'. Final Bill Amount: ₹{final_total:,.2f}. Stock updated to {new_stock} {unit}."
    return df, sale_record, msg

def refill_product_stock(df_current: pd.DataFrame, product_id: str, refill_qty: float, notes: str = None) -> tuple[pd.DataFrame, dict, str]:
    """
    Refills stock for a product by adding refill_qty to Current_Stock.
    Validates non-negative quantity and records STOCK_REFILL transaction.
    Returns (updated_df, transaction_record, success_message).
    """
    df = df_current.copy()
    matched = df[df["Product_ID"] == product_id]

    if matched.empty:
        raise ValueError(f"Product ID '{product_id}' not found in inventory.")

    try:
        val_qty = float(refill_qty)
        if val_qty <= 0:
            raise ValueError("Refill quantity must be greater than zero.")
    except (ValueError, TypeError):
        raise ValueError("Refill quantity must be a valid positive number.")

    idx = matched.index[0]
    p_name = str(df.loc[idx, "Product_Name"])
    category = str(df.loc[idx, "Category"]) if "Category" in df.columns else "General"
    unit = str(df.loc[idx, "Unit"]) if "Unit" in df.columns else "units"
    prev_stock = float(df.loc[idx, "Current_Stock"])

    new_stock = prev_stock + val_qty
    df.loc[idx, "Current_Stock"] = new_stock

    notes_str = notes if notes else f"Stock Refill: Added +{val_qty} {unit} to stock ({prev_stock} -> {new_stock})"

    log_transaction(
        trans_type="STOCK_REFILL",
        product_id=product_id,
        product_name=p_name,
        category=category,
        qty_change=val_qty,
        unit=unit,
        prev_stock=prev_stock,
        new_stock=new_stock,
        notes=notes_str
    )

    msg = f"Successfully refilled {val_qty} {unit} for '{p_name}'. Stock updated from {prev_stock} {unit} to {new_stock} {unit}."
    res_rec = df.loc[idx].to_dict()
    return df, res_rec, msg

def refill_batch_products(df_current: pd.DataFrame, refill_list: list) -> tuple[pd.DataFrame, int, list]:
    """
    Processes batch stock refills for a list of items: [{'product_id': '...', 'refill_qty': 40.0}]
    Returns (updated_df, success_count, error_messages).
    """
    df = df_current.copy()
    success_count = 0
    errors = []

    for item in refill_list:
        p_id = item.get("product_id")
        q_refill = item.get("refill_qty")
        try:
            df, _, _ = refill_product_stock(df, p_id, q_refill, notes="Batch Stock Refill")
            success_count += 1
        except Exception as e:
            errors.append(f"Failed to refill Product ID {p_id}: {e}")

    return df, success_count, errors

