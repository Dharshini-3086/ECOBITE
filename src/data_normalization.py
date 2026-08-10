"""
Data Normalization & Validation Layer for ECOBITE
Handles column alias mapping, data type conversions, date parsing, field validation,
and product identity matching across single and batch inventory inputs.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date

# Standard expected columns for ECOBITE ML & Inventory Pipeline
STANDARD_COLUMNS = [
    "Product_ID", "Product_Name", "Category", "Current_Stock", "Unit",
    "Historical_Sales", "Daily_Sales", "Selling_Price", "Previous_Discount",
    "Sales_During_Discount", "Season", "Month", "Days_to_Expiry",
    "Supplier_Lead_Time", "Promotion", "Waste_Label"
]

# Alias map for flexible column matching in CSV uploads
COLUMN_ALIAS_MAP = {
    # Product ID
    "product_id": "Product_ID", "id": "Product_ID", "item_id": "Product_ID", "sku": "Product_ID", "code": "Product_ID",
    # Product Name
    "product_name": "Product_Name", "product": "Product_Name", "item_name": "Product_Name", "item": "Product_Name", "name": "Product_Name",
    # Category
    "category": "Category", "cat": "Category", "department": "Category", "type": "Category",
    # Stock / Quantity
    "current_stock": "Current_Stock", "stock": "Current_Stock", "quantity": "Current_Stock", "qty": "Current_Stock", "inventory": "Current_Stock", "stock_quantity": "Current_Stock",
    # Unit
    "unit": "Unit", "units": "Unit", "uom": "Unit", "unit_of_measure": "Unit",
    # Selling Price
    "selling_price": "Selling_Price", "price": "Selling_Price", "unit_price": "Selling_Price", "mrp": "Selling_Price", "cost_price": "Selling_Price",
    # Historical Sales
    "historical_sales": "Historical_Sales", "past_sales": "Historical_Sales", "monthly_sales": "Historical_Sales", "sales_30d": "Historical_Sales",
    # Daily Sales
    "daily_sales": "Daily_Sales", "sales_velocity": "Daily_Sales", "avg_daily_sales": "Daily_Sales",
    # Expiry Date / Days to Expiry
    "expiry_date": "Days_to_Expiry", "expiry": "Days_to_Expiry", "expiration_date": "Days_to_Expiry", "days_to_expiry": "Days_to_Expiry", "exp_date": "Days_to_Expiry",
    # Previous Discount
    "previous_discount": "Previous_Discount", "prev_discount": "Previous_Discount", "discount": "Previous_Discount", "discount_pct": "Previous_Discount",
    # Sales During Discount
    "sales_during_discount": "Sales_During_Discount", "discount_sales": "Sales_During_Discount",
    # Season
    "season": "Season", "peak_season": "Season",
    # Month
    "month": "Month",
    # Supplier Lead Time
    "supplier_lead_time": "Supplier_Lead_Time", "lead_time": "Supplier_Lead_Time", "lead_days": "Supplier_Lead_Time",
    # Promotion
    "promotion": "Promotion", "promo": "Promotion", "on_promo": "Promotion"
}

# Known ambiguous column names that should NOT be blindly mapped
AMBIGUOUS_COLUMNS = {"amount", "value", "rate", "total", "count", "num", "date", "period", "level"}

def calculate_days_to_expiry(expiry_input) -> int:
    """
    Computes Days_to_Expiry from an expiry date string/date object or integer.
    """
    if pd.isna(expiry_input) or expiry_input is None:
        return 15 # Reasonable safe default
        
    if isinstance(expiry_input, (int, float, np.integer, np.floating)):
        return max(0, int(expiry_input))
        
    if isinstance(expiry_input, (datetime, date)):
        today = date.today()
        exp_d = expiry_input.date() if isinstance(expiry_input, datetime) else expiry_input
        return (exp_d - today).days

    # Parse string date formats
    str_val = str(expiry_input).strip()
    if str_val.isdigit() or (str_val.startswith('-') and str_val[1:].isdigit()):
        return int(str_val)
        
    date_formats = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%b-%Y", "%d %b %Y"]
    for fmt in date_formats:
        try:
            dt = datetime.strptime(str_val, fmt).date()
            today = date.today()
            return (dt - today).days
        except ValueError:
            pass

    try:
        dt = pd.to_datetime(str_val).date()
        return (dt - date.today()).days
    except Exception:
        return 15

def validate_single_product_input(data: dict) -> tuple[bool, list[str]]:
    """
    Validates single product insertion form data.
    Returns (is_valid, list_of_error_messages).
    """
    errors = []

    # Required field existence checks
    prod_name = str(data.get("Product_Name", "")).strip()
    if not prod_name:
        errors.append("Product Name is required.")

    category = str(data.get("Category", "")).strip()
    if not category:
        errors.append("Category is required.")

    # Stock validation
    stock = data.get("Current_Stock", None)
    if stock is None or str(stock).strip() == "":
        errors.append("Current Stock is required.")
    else:
        try:
            val_stock = float(stock)
            if val_stock < 0:
                errors.append("Stock quantity cannot be negative.")
        except (ValueError, TypeError):
            errors.append("Current Stock must be a valid number.")

    # Price validation
    price = data.get("Selling_Price", None)
    if price is None or str(price).strip() == "":
        errors.append("Selling Price is required.")
    else:
        try:
            val_price = float(price)
            if val_price <= 0:
                errors.append("Selling price must be greater than zero.")
        except (ValueError, TypeError):
            errors.append("Selling Price must be a valid number.")

    # Expiry Date / Days validation
    days_to_exp = data.get("Days_to_Expiry", None)
    expiry_date = data.get("Expiry_Date", None)
    if days_to_exp is None and expiry_date is None:
        errors.append("Expiry Date is required.")
    elif days_to_exp is not None:
        try:
            if float(days_to_exp) < 0:
                errors.append("Expiry date cannot be in the past.")
        except (ValueError, TypeError):
            errors.append("Days to Expiry must be a valid number.")

    # Historical / Daily Sales validation
    hist_sales = data.get("Historical_Sales", None)
    daily_sales = data.get("Daily_Sales", None)
    if hist_sales is None and daily_sales is None:
        errors.append("Historical Sales is required.")
    else:
        if hist_sales is not None:
            try:
                if float(hist_sales) < 0:
                    errors.append("Historical Sales cannot be negative.")
            except (ValueError, TypeError):
                errors.append("Historical Sales must be a valid number.")

    # Discount validation
    prev_disc = data.get("Previous_Discount", 0.0)
    if prev_disc is not None and str(prev_disc).strip() != "":
        try:
            val_disc = float(prev_disc)
            if val_disc < 0 or val_disc > 100:
                errors.append("Discount percentage must be between 0 and 100.")
        except (ValueError, TypeError):
            errors.append("Previous Discount must be a valid number.")

    # Supplier Lead Time validation
    lead_time = data.get("Supplier_Lead_Time", 5)
    if lead_time is not None and str(lead_time).strip() != "":
        try:
            if float(lead_time) < 0:
                errors.append("Supplier Lead Time cannot be negative.")
        except (ValueError, TypeError):
            errors.append("Supplier Lead Time must be a valid number.")

    is_valid = len(errors) == 0
    return is_valid, errors

def normalize_inventory_record(data: dict) -> dict:
    """
    Standardizes a single product dictionary to match the ECOBITE ML schema.
    """
    normalized = {}

    # Product ID
    p_id = str(data.get("Product_ID", "")).strip()
    if not p_id or p_id == "nan":
        p_name_clean = str(data.get("Product_Name", "PRD")).upper().replace(" ", "")[:4]
        p_id = f"PRD-{p_name_clean}-{int(datetime.now().timestamp()) % 10000}"
    normalized["Product_ID"] = p_id

    normalized["Product_Name"] = str(data.get("Product_Name", "Unknown Product")).strip()
    normalized["Category"] = str(data.get("Category", "General")).strip()
    normalized["Unit"] = str(data.get("Unit", "units")).strip().lower()

    # Numbers
    normalized["Current_Stock"] = float(data.get("Current_Stock", 0.0))
    normalized["Selling_Price"] = float(data.get("Selling_Price", 100.0))

    # Sales
    hist_sales = data.get("Historical_Sales", None)
    daily_sales = data.get("Daily_Sales", None)

    if hist_sales is not None:
        normalized["Historical_Sales"] = float(hist_sales)
        normalized["Daily_Sales"] = float(daily_sales) if daily_sales is not None else round(float(hist_sales) / 30.0, 1)
    elif daily_sales is not None:
        normalized["Daily_Sales"] = float(daily_sales)
        normalized["Historical_Sales"] = float(round(float(daily_sales) * 30.0, 0))
    else:
        normalized["Historical_Sales"] = 100.0
        normalized["Daily_Sales"] = 3.3

    normalized["Previous_Discount"] = float(data.get("Previous_Discount", 0.0))
    normalized["Sales_During_Discount"] = float(data.get("Sales_During_Discount", normalized["Daily_Sales"]))

    # Expiry
    if "Days_to_Expiry" in data and data["Days_to_Expiry"] is not None and str(data["Days_to_Expiry"]).strip() != "":
        normalized["Days_to_Expiry"] = int(max(0, calculate_days_to_expiry(data["Days_to_Expiry"])))
    elif "Expiry_Date" in data and data["Expiry_Date"] is not None and str(data["Expiry_Date"]).strip() != "":
        normalized["Days_to_Expiry"] = int(max(0, calculate_days_to_expiry(data["Expiry_Date"])))
    else:
        normalized["Days_to_Expiry"] = 15

    # Season & Month
    normalized["Season"] = str(data.get("Season", "All-Season")).strip()
    normalized["Month"] = int(data.get("Month", date.today().month))
    normalized["Supplier_Lead_Time"] = int(data.get("Supplier_Lead_Time", 5))
    normalized["Promotion"] = int(data.get("Promotion", 0))
    normalized["Waste_Label"] = int(data.get("Waste_Label", 0))

    return normalized

def normalize_inventory_dataframe(df_raw: pd.DataFrame, custom_mappings: dict = None) -> tuple[pd.DataFrame, dict, list]:
    """
    Normalizes an uploaded CSV DataFrame into standard ECOBITE schema.
    Returns (normalized_df, column_mapping_used, unmappable_columns).
    """
    df = df_raw.copy()
    mapping_used = {}
    unmappable = []

    # Map column names
    renamed_cols = {}
    for col in df.columns:
        col_clean = str(col).strip()
        col_lower = col_clean.lower().replace(" ", "_")

        if custom_mappings and col_clean in custom_mappings:
            target_col = custom_mappings[col_clean]
            renamed_cols[col] = target_col
            mapping_used[col_clean] = target_col
        elif col_clean in STANDARD_COLUMNS:
            renamed_cols[col] = col_clean
            mapping_used[col_clean] = col_clean
        elif col_lower in COLUMN_ALIAS_MAP:
            target_col = COLUMN_ALIAS_MAP[col_lower]
            renamed_cols[col] = target_col
            mapping_used[col_clean] = target_col
        elif col_lower in AMBIGUOUS_COLUMNS:
            unmappable.append(col_clean)
        else:
            unmappable.append(col_clean)

    df = df.rename(columns=renamed_cols)

    # Ensure required fields or construct defaults
    records = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        norm_dict = normalize_inventory_record(row_dict)
        records.append(norm_dict)

    df_norm = pd.DataFrame(records)
    return df_norm, mapping_used, unmappable

def is_same_product(p1: dict, p2: dict) -> bool:
    """
    Checks whether two product records represent the exact same product identity.
    Uses Product_ID first, or fallback to normalized (Product_Name + Category + Unit).
    """
    id1 = str(p1.get("Product_ID", "")).strip().upper()
    id2 = str(p2.get("Product_ID", "")).strip().upper()

    if id1 and id2 and id1 == id2 and not id1.startswith("PRD-TEMP"):
        return True

    name1 = str(p1.get("Product_Name", "")).strip().lower().replace(" ", "")
    name2 = str(p2.get("Product_Name", "")).strip().lower().replace(" ", "")

    cat1 = str(p1.get("Category", "")).strip().lower()
    cat2 = str(p2.get("Category", "")).strip().lower()

    unit1 = str(p1.get("Unit", "units")).strip().lower()
    unit2 = str(p2.get("Unit", "units")).strip().lower()

    return (name1 == name2) and (cat1 == cat2) and (unit1 == unit2)

def calculate_refill_metrics(row) -> dict:
    """
    Computes low-stock detection metrics, reorder points, safety stock,
    recommended refill quantities, stock statuses, and urgency levels for a product.
    """
    row_d = row.to_dict() if isinstance(row, pd.Series) else row

    stock = float(row_d.get("Current_Stock", 0.0))
    daily_sales = float(row_d.get("Daily_Sales", 1.0))
    pred_demand = float(row_d.get("Predicted_Demand", daily_sales * 15.0))
    days_to_exp = max(1.0, float(row_d.get("Days_to_Expiry", 15)))

    # Daily demand estimation
    daily_demand = max(0.1, pred_demand / days_to_exp) if pred_demand > 0 else max(0.1, daily_sales)

    # Lead time
    lead_time = int(row_d.get("Supplier_Lead_Time", 5))

    # Min stock & safety stock
    min_stock_custom = row_d.get("Minimum_Stock", None)
    if min_stock_custom is not None and not pd.isna(min_stock_custom):
        min_stock = float(min_stock_custom)
    else:
        min_stock = float(max(10, int(daily_demand * 3)))

    lead_time_demand = daily_demand * lead_time
    safety_stock = max(min_stock, lead_time_demand * 0.5)

    # Reorder Point
    reorder_point = round(lead_time_demand + safety_stock, 1)

    # Target stock level for refill
    target_stock = round(max(reorder_point * 2.5, min_stock * 3, 50.0), 0)

    # Recommended Refill Qty
    recommended_refill = float(max(0.0, round(target_stock - stock, 0)))

    # Stock Status determination
    if stock == 0:
        stock_status = "OUT OF STOCK"
    elif stock < reorder_point:
        stock_status = "REORDER REQUIRED"
    elif stock <= reorder_point * 1.3:
        stock_status = "LOW STOCK"
    else:
        stock_status = "HEALTHY STOCK"

    # Urgency determination
    if stock == 0 or stock <= reorder_point * 0.5:
        urgency = "REFILL IMMEDIATELY"
    elif stock < reorder_point:
        urgency = "REFILL SOON"
    elif stock <= reorder_point * 1.3:
        urgency = "PLAN REFILL"
    else:
        urgency = "NORMAL"

    supplier = str(row_d.get("Supplier", "ABC Supermarket Distributors"))

    return {
        "Product_ID": str(row_d.get("Product_ID", "")),
        "Product_Name": str(row_d.get("Product_Name", "")),
        "Category": str(row_d.get("Category", "")),
        "Unit": str(row_d.get("Unit", "units")),
        "Current_Stock": stock,
        "Minimum_Stock": round(min_stock, 1),
        "Daily_Demand": round(daily_demand, 1),
        "Supplier_Lead_Time": lead_time,
        "Reorder_Point": reorder_point,
        "Target_Stock": target_stock,
        "Recommended_Refill": recommended_refill if stock < reorder_point * 1.3 else 0.0,
        "Stock_Status": stock_status,
        "Urgency": urgency,
        "Supplier": supplier
    }

def compute_refill_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes stock refill metrics across an entire inventory DataFrame.
    """
    metrics_list = []
    for _, row in df.iterrows():
        metrics_list.append(calculate_refill_metrics(row))
    return pd.DataFrame(metrics_list)

