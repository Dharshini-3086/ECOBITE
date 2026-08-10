"""
ECOBITE - Streamlit Decision-Support Dashboard
AI-Based Supermarket Inventory Waste Prediction, Dynamic Discount & Retail Management System
"""

import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import io
from datetime import datetime, date
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.prediction import EcoBitePredictor
from src.utils import (
    apply_custom_css, get_risk_badge_html, get_inventory_status_badge_html,
    fuzzy_search_dataframe, get_stock_status_badge_html, get_urgency_badge_html
)
from src.discount_engine import calculate_dynamic_discount
from src.recommendation_engine import generate_inventory_recommendation
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

# 1. Page Configuration & Setup
st.set_page_config(
    page_title="ECOBITE - Inventory Waste & Dynamic Discount AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SAMPLE_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "sample_upload_test.csv")

@st.cache_resource
def get_predictor():
    return EcoBitePredictor(models_dir=MODELS_DIR)

predictor = get_predictor()

# Check models existence
if not predictor.is_loaded:
    st.warning("⚠️ Machine learning models not found in `models/`. Training models now...")
    with st.spinner("Training Random Forest Waste & Demand models..."):
        from src.data.raw.generate_synthetic_data import main as gen_data
        from src.train_waste_model import train_waste_classification_model
        from src.train_demand_model import train_demand_regression_model
        
        gen_data()
        train_waste_classification_model()
        train_demand_regression_model()
        predictor.load_models()
    st.success("✅ Models trained and loaded successfully!")

# Initialize session state for active inventory dataset
def refresh_predictions():
    df_raw = load_inventory_dataframe()
    df_pred = predictor.predict_dataframe(df_raw)
    st.session_state["df_inventory"] = df_pred

if "df_inventory" not in st.session_state:
    refresh_predictions()

df_active = st.session_state["df_inventory"]

# 2. Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/000000/supermarket.png", width=70)
st.sidebar.title("ECOBITE AI System")
st.sidebar.caption("Supermarket Waste & Retail Decision Engine")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Dashboard Overview",
        "📦 Inventory Management",
        "🔍 Single Product Deep-Dive",
        "📋 Inventory Action Center",
        "📈 Model Performance & ML Analytics",
        "⚙️ What-If Analysis Simulator"
    ]
)

st.sidebar.markdown("---")
currency_sym = st.sidebar.selectbox("Currency", ["₹ (INR)", "$ (USD)", "€ (EUR)"], index=0)[0]
st.sidebar.caption(f"Active Products: **{len(df_active)}** | Stock Total: **{int(df_active['Current_Stock'].sum()):,}**")

# Header
st.title("🛒 ECOBITE - AI Supermarket Inventory & Waste Decision Engine")
st.caption("Predict Waste → Forecast Demand → Calculate Excess → Dynamic Discount → Manage Inventory → Sales & Billing")
st.markdown("---")

# ----------------------------------------------------
# PAGE 1: OVERVIEW DASHBOARD
# ----------------------------------------------------
if nav_choice == "📊 Dashboard Overview":
    st.subheader("📊 Executive Overview & Inventory Risk Metrics")

    # Top KPI Metrics
    total_products = len(df_active)
    total_inventory = int(df_active["Current_Stock"].sum())
    high_critical_risk = len(df_active[df_active["Risk_Level"].isin(["HIGH", "CRITICAL"])])
    total_excess = int(df_active["Excess_Inventory"].sum())
    potential_savings = df_active["Potential_Savings_INR"].sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total SKU Products", f"{total_products:,}")
    col2.metric("Total Stock Units", f"{total_inventory:,}")
    col3.metric("High/Critical Risk SKUs", f"{high_critical_risk:,}", delta=f"{high_critical_risk/max(1,total_products)*100:.1f}% of total", delta_color="inverse")
    col4.metric("Total Excess Units", f"{total_excess:,}")
    # Calculate refill metrics for dashboard alert
    df_refill_dash = compute_refill_dataframe(df_active)
    out_cnt = len(df_refill_dash[df_refill_dash["Stock_Status"] == "OUT OF STOCK"])
    reorder_cnt = len(df_refill_dash[df_refill_dash["Stock_Status"] == "REORDER REQUIRED"])
    if out_cnt > 0 or reorder_cnt > 0:
        st.warning(f"🔄 **Stock Refill Alert:** **{out_cnt}** products are Out of Stock and **{reorder_cnt}** products are below their reorder threshold. Go to `📦 Inventory Management` $\rightarrow$ `🔄 Stock Refill` to execute refills.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row 1
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("##### 🎯 Waste Risk Level Distribution")
        risk_counts = df_active["Risk_Level"].value_counts().reset_index()
        risk_counts.columns = ["Risk_Level", "Count"]
        color_map = {"LOW": "#10b981", "MEDIUM": "#f59e0b", "HIGH": "#ef4444", "CRITICAL": "#7f1d1d"}
        fig_donut = px.pie(
            risk_counts, values="Count", names="Risk_Level",
            color="Risk_Level", color_discrete_map=color_map,
            hole=0.45
        )
        fig_donut.update_layout(template="plotly_dark", margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        st.markdown("##### 📦 Current Stock vs Predicted Demand by Category")
        cat_agg = df_active.groupby("Category")[["Current_Stock", "Predicted_Demand"]].sum().reset_index()
        fig_bar = px.bar(
            cat_agg, x="Category", y=["Current_Stock", "Predicted_Demand"],
            barmode="group",
            labels={"value": "Units", "variable": "Metric"},
            color_discrete_sequence=["#3b82f6", "#10b981"]
        )
        fig_bar.update_layout(template="plotly_dark", margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    # Charts Row 2
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("##### ⏳ Days to Expiry vs Waste Risk Probability")
        fig_scatter = px.scatter(
            df_active, x="Days_to_Expiry", y="Waste_Probability",
            color="Risk_Level", size="Current_Stock",
            hover_data=["Product_Name", "Category", "Excess_Inventory"],
            color_discrete_map=color_map,
            labels={"Days_to_Expiry": "Days Remaining to Expiry", "Waste_Probability": "Waste Risk Probability"}
        )
        fig_scatter.update_layout(template="plotly_dark", margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with c4:
        st.markdown("##### 🚨 Top 5 Products Requiring Urgent Action")
        urgent_df = df_active.sort_values(by="Waste_Probability", ascending=False).head(5)
        fig_urgent = px.bar(
            urgent_df, x="Product_Name", y="Excess_Inventory",
            color="Risk_Level", text="Recommended_Discount",
            color_discrete_map=color_map,
            labels={"Excess_Inventory": "Excess Stock Units", "Product_Name": "Product"}
        )
        fig_urgent.update_traces(texttemplate='%{text}% Disc', textposition='outside')
        fig_urgent.update_layout(template="plotly_dark", margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_urgent, use_container_width=True)

# ----------------------------------------------------
# PAGE 2: INVENTORY MANAGEMENT (INSERT, BATCH, SELL, LIST)
# ----------------------------------------------------
elif nav_choice == "📦 Inventory Management":
    st.subheader("📦 Supermarket Inventory Management Hub")

    inv_tab1, inv_tab2, inv_tab3, inv_tab4, inv_tab5, inv_tab6 = st.tabs([
        "📋 Inventory List & Search",
        "➕ Add Single Product",
        "📁 Batch CSV Upload",
        "🧾 Sell Product / Billing",
        "📜 Transaction Audit Logs",
        "🔄 Stock Refill"
    ])

    # ----------------------------------------------------
    # TAB 1: INVENTORY LIST & SEARCH
    # ----------------------------------------------------
    with inv_tab1:
        st.markdown("#### 🔍 Master Inventory Search & Filter")
        
        col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
        with col_s1:
            all_prods_list = ["-- All Products --"] + sorted(list(df_active["Product_Name"].unique()))
            selected_sug_dropdown = st.selectbox(
                "💡 Quick Autocomplete Suggestion",
                all_prods_list,
                index=0,
                key="search_autocomplete_sel"
            )
            
            if selected_sug_dropdown != "-- All Products --":
                search_query = selected_sug_dropdown
            else:
                if "inv_search_str_val" not in st.session_state:
                    st.session_state["inv_search_str_val"] = ""
                search_query = st.text_input(
                    "🔍 Type to Search (e.g. su, toma, milk)...",
                    value=st.session_state["inv_search_str_val"],
                    key="inv_search_q_input"
                )
                st.session_state["inv_search_str_val"] = search_query

                # Live Auto-Suggestion Pills below text input
                if search_query.strip():
                    matched_sug = fuzzy_search_dataframe(df_active, search_query, ["Product_Name", "Product_ID", "Category"])
                    sug_names = matched_sug["Product_Name"].unique()[:5]
                    if len(sug_names) > 0:
                        st.caption("💡 **Auto-Suggestions (Click to apply):**")
                        sug_cols = st.columns(min(5, len(sug_names)))
                        for idx, s_name in enumerate(sug_names):
                            if sug_cols[idx].button(f"🏷️ {s_name}", key=f"sug_pill_btn_{idx}"):
                                st.session_state["inv_search_str_val"] = s_name
                                st.rerun()

        with col_s2:
            risk_filter = st.multiselect("Filter Risk", ["LOW", "MEDIUM", "HIGH", "CRITICAL"], default=["LOW", "MEDIUM", "HIGH", "CRITICAL"])
        with col_s3:
            cat_options = list(df_active["Category"].unique())
            cat_filter = st.multiselect("Filter Category", cat_options, default=cat_options)

        # Apply search and filters
        filtered_df = df_active[
            (df_active["Risk_Level"].isin(risk_filter)) &
            (df_active["Category"].isin(cat_filter))
        ]
        if search_query:
            filtered_df = fuzzy_search_dataframe(filtered_df, search_query, ["Product_Name", "Product_ID", "Category"])

        st.markdown(f"**Showing {len(filtered_df)} of {len(df_active)} products:**")

        disp_cols = [
            "Product_ID", "Product_Name", "Category", "Current_Stock", "Unit",
            "Days_to_Expiry", "Selling_Price", "Predicted_Demand", "Excess_Inventory",
            "Waste_Probability", "Risk_Level", "Recommended_Discount", "Recommended_Price", "Recommended_Action"
        ]

        display_table = filtered_df[disp_cols].copy()
        display_table["Waste_Probability"] = display_table["Waste_Probability"].apply(lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "")
        display_table["Recommended_Discount"] = display_table["Recommended_Discount"].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "")
        display_table["Selling_Price"] = display_table["Selling_Price"].apply(lambda x: f"{currency_sym}{float(x):,.2f}" if pd.notna(x) and str(x) != "" else "")
        display_table["Recommended_Price"] = display_table["Recommended_Price"].apply(lambda x: f"{currency_sym}{float(x):,.2f}" if pd.notna(x) and str(x) != "" else "")

        st.dataframe(display_table, use_container_width=True, height=450)

    # ----------------------------------------------------
    # TAB 2: ADD SINGLE PRODUCT FORM
    # ----------------------------------------------------
    with inv_tab2:
        st.markdown("#### ➕ Add Product to Supermarket Inventory")
        st.caption("Fill in product information to calculate demand, waste risk, and dynamic discounts automatically.")

        # Multi-step layout using expandable sections
        with st.form("add_product_form", clear_on_submit=False):
            st.markdown('<div class="step-header">Step 1: 📦 Product & Category Info</div>', unsafe_allow_html=True)
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                p_name = st.text_input("Product Name *", value="", placeholder="e.g. Tomato, Potato, Milk")
            with f_col2:
                existing_cats = ["Produce", "Dairy", "Bakery", "Beverage", "Personal Care", "Pantry", "Frozen", "Household", "Vegetables", "Other"]
                p_cat = st.selectbox("Category *", existing_cats, index=0)
            with f_col3:
                p_unit = st.selectbox("Unit *", ["kg", "units", "L", "g", "ml", "pack", "box"], index=0)

            p_id_custom = st.text_input("Product ID (Optional - Auto-generated if blank)", value="", placeholder="e.g. PRD-1025")

            st.markdown('<div class="step-header">Step 2: 📊 Inventory & Sales History</div>', unsafe_allow_html=True)
            f_col4, f_col5, f_col6 = st.columns(3)
            with f_col4:
                p_stock = st.number_input("Current Stock Quantity *", min_value=0.0, value=15.0, step=1.0)
            with f_col5:
                p_hist_sales = st.number_input("Historical Sales (Last 30 Days) *", min_value=0.0, value=90.0, step=5.0)
            with f_col6:
                p_daily_sales = st.number_input("Average Daily Sales Velocity (units/day)", min_value=0.1, value=round(90.0/30.0, 1), step=0.5)

            st.markdown('<div class="step-header">Step 3: 📅 Expiry Date & Seasonality</div>', unsafe_allow_html=True)
            f_col7, f_col8, f_col9 = st.columns(3)
            with f_col7:
                default_exp_date = date.today() + pd.Timedelta(days=15)
                p_expiry_date = st.date_input("Expiry Date *", value=default_exp_date)
                calc_days_exp = (p_expiry_date - date.today()).days
                st.caption(f"💡 Calculated Days to Expiry: **{calc_days_exp} days**")
            with f_col8:
                p_season = st.selectbox("Season *", ["Summer", "Winter", "Monsoon", "Spring"], index=0)
            with f_col9:
                p_month = st.slider("Month (1-12)", min_value=1, max_value=12, value=p_expiry_date.month)

            st.markdown('<div class="step-header">Step 4: 💰 Pricing & Discount Sensitivity</div>', unsafe_allow_html=True)
            f_col10, f_col11, f_col12 = st.columns(3)
            with f_col10:
                p_price = st.number_input("Selling Price (per unit) *", min_value=0.1, value=80.0, step=5.0)
            with f_col11:
                p_prev_disc = st.number_input("Previous Discount (%)", min_value=0.0, max_value=100.0, value=0.0, step=5.0)
            with f_col12:
                p_disc_sales = st.number_input("Sales Velocity During Discount", min_value=0.1, value=round(p_daily_sales * 1.2, 1), step=0.5)

            st.markdown('<div class="step-header">Step 5: 🚚 Supplier & Promotion</div>', unsafe_allow_html=True)
            f_col13, f_col14 = st.columns(2)
            with f_col13:
                p_lead_time = st.number_input("Supplier Lead Time (Days)", min_value=0, value=5, step=1)
            with f_col14:
                p_promo = st.selectbox("Active Promotion?", ["No (0)", "Yes (1)"], index=0)
                promo_val = 1 if "Yes" in p_promo else 0

            st.markdown("---")
            submit_add = st.form_submit_button("✅ Review & Add Product to Inventory", use_container_width=True)

        if submit_add:
            # Prepare payload dictionary
            raw_input = {
                "Product_ID": p_id_custom if p_id_custom.strip() else None,
                "Product_Name": p_name,
                "Category": p_cat,
                "Unit": p_unit,
                "Current_Stock": p_stock,
                "Historical_Sales": p_hist_sales,
                "Daily_Sales": p_daily_sales,
                "Selling_Price": p_price,
                "Expiry_Date": p_expiry_date,
                "Days_to_Expiry": calc_days_exp,
                "Previous_Discount": p_prev_disc,
                "Sales_During_Discount": p_disc_sales,
                "Season": p_season,
                "Month": p_month,
                "Supplier_Lead_Time": p_lead_time,
                "Promotion": promo_val
            }

            # 1. Field validation
            is_valid, validation_errors = validate_single_product_input(raw_input)
            if not is_valid:
                st.error("❌ Cannot add product. Please complete or fix the following required fields:")
                for err in validation_errors:
                    st.markdown(f"- ⚠️ {err}")
            else:
                st.session_state["pending_product_input"] = raw_input
                st.success("✅ Form inputs validated!")

        # Confirmation & Merge Dialog
        if "pending_product_input" in st.session_state and st.session_state["pending_product_input"] is not None:
            pend = st.session_state["pending_product_input"]
            norm_pend = normalize_inventory_record(pend)

            # Check if product already exists
            existing_match = None
            for _, row in df_active.iterrows():
                if is_same_product(row.to_dict(), norm_pend):
                    existing_match = row
                    break

            st.markdown("---")
            st.markdown("### 📋 Review Inventory Entry")

            if existing_match is not None:
                curr_qty = float(existing_match["Current_Stock"])
                new_qty = float(norm_pend["Current_Stock"])
                final_qty = curr_qty + new_qty
                unit_str = norm_pend["Unit"]

                st.warning(f"⚠️ **'{norm_pend['Product_Name']}' already exists in inventory.**")
                
                m_c1, m_c2, m_c3 = st.columns(3)
                m_c1.metric("Current Stock", f"{curr_qty} {unit_str}")
                m_c2.metric("New Stock to Add", f"+{new_qty} {unit_str}")
                m_c3.metric("Final Merged Stock", f"={final_qty} {unit_str}")

                st.markdown(f"**Do you want to MERGE this inventory quantity?** (`{curr_qty} + {new_qty} = {final_qty} {unit_str}`)")

                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    if st.button("❌ Cancel Insert", use_container_width=True):
                        st.session_state["pending_product_input"] = None
                        st.rerun()
                with b_col2:
                    if st.button("🔄 Merge & Update Inventory", use_container_width=True):
                        try:
                            df_new, res_record, msg = merge_or_add_product(df_active, norm_pend)
                            save_inventory_dataframe(df_new)
                            refresh_predictions()
                            st.session_state["pending_product_input"] = None
                            st.success(f"✅ {msg}")
                            st.rerun()
                        except Exception as ex:
                            st.error(f"❌ Error merging product: {ex}")
            else:
                unit_str = norm_pend["Unit"]
                st.info(f"🆕 **Creating new inventory record for '{norm_pend['Product_Name']}' ({norm_pend['Current_Stock']} {unit_str}).**")

                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    if st.button("❌ Cancel", use_container_width=True):
                        st.session_state["pending_product_input"] = None
                        st.rerun()
                with b_col2:
                    if st.button("✅ Confirm & Add Product", use_container_width=True):
                        try:
                            df_new, res_record, msg = merge_or_add_product(df_active, norm_pend)
                            save_inventory_dataframe(df_new)
                            refresh_predictions()
                            st.session_state["pending_product_input"] = None
                            st.success(f"✅ {msg}")
                            st.rerun()
                        except Exception as ex:
                            st.error(f"❌ Error adding product: {ex}")

    # ----------------------------------------------------
    # TAB 3: BATCH CSV UPLOAD
    # ----------------------------------------------------
    with inv_tab3:
        st.markdown("#### 📁 Upload Batch Supermarket CSV")
        st.caption("Upload a CSV file containing multiple products. The system normalizes headers, validates fields, and merges quantities automatically.")

        c_up1, c_up2 = st.columns([2, 1])
        with c_up1:
            uploaded_file = st.file_uploader("Choose CSV File to Upload", type=["csv"], key="batch_csv_up")
        with c_up2:
            st.markdown("##### 📥 Sample Upload File")
            if os.path.exists(SAMPLE_PATH):
                sample_df = pd.read_csv(SAMPLE_PATH)
                st.download_button(
                    label="📥 Download Sample CSV",
                    data=sample_df.to_csv(index=False).encode('utf-8'),
                    file_name="sample_supermarket_upload.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        if uploaded_file is not None:
            try:
                df_raw_up = pd.read_csv(uploaded_file)
                st.markdown("##### 📄 Raw CSV Data Preview:")
                st.dataframe(df_raw_up.head(5), use_container_width=True)

                # Run normalization
                df_norm_up, map_used, unmappable = normalize_inventory_dataframe(df_raw_up)

                st.markdown("---")
                st.markdown("#### 📊 Dataset Validation & Mapping Preview")
                
                v_col1, v_col2, v_col3 = st.columns(3)
                v_col1.metric("Rows Detected", f"{len(df_raw_up)}")
                v_col2.metric("Valid Rows Processed", f"{len(df_norm_up)}")
                v_col3.metric("Unmapped Columns", f"{len(unmappable)}")

                st.markdown("##### 🔍 Column Mappings Applied:")
                st.json(map_used)

                if unmappable:
                    st.warning(f"⚠️ Column(s) could not be automatically mapped: `{unmappable}`. Default values will be used for optional fields.")

                st.markdown("##### 🎯 Normalized Batch Preview:")
                st.dataframe(df_norm_up.head(5), use_container_width=True)

                b_up1, b_up2 = st.columns(2)
                with b_up1:
                    if st.button("❌ Cancel Batch Upload", use_container_width=True):
                        st.rerun()
                with b_up2:
                    if st.button("🚀 Process & Merge Batch into Inventory", use_container_width=True):
                        with st.spinner("Processing batch products and updating ML pipeline..."):
                            df_current = df_active.copy()
                            added_count = 0
                            merged_count = 0
                            
                            for _, row in df_norm_up.iterrows():
                                row_d = row.to_dict()
                                try:
                                    df_current, res_rec, msg = merge_or_add_product(df_current, row_d)
                                    if "Merged" in msg:
                                        merged_count += 1
                                    else:
                                        added_count += 1
                                except Exception as e:
                                    pass

                            save_inventory_dataframe(df_current)
                            refresh_predictions()
                            st.success(f"✅ Batch Processing Complete! Added {added_count} new products, merged {merged_count} existing products.")
                            st.rerun()

            except Exception as e:
                st.error(f"❌ Error reading uploaded file: {e}")

    # ----------------------------------------------------
    # TAB 4: SELL PRODUCT / BILLING
    # ----------------------------------------------------
    with inv_tab4:
        st.markdown("#### 🧾 Create Bill & Sell Product")
        st.caption("Select a product to sell, enter the sold quantity, apply recommended discounts, and update inventory instantly.")

        sel_prod_str = st.selectbox(
            "Select Product to Sell:",
            df_active["Product_ID"] + " - " + df_active["Product_Name"] + " (Stock: " + df_active["Current_Stock"].astype(int).astype(str) + " " + df_active.get("Unit", "units") + ")",
            index=0
        )
        
        sel_prod_id = sel_prod_str.split(" - ")[0]
        prod_row = df_active[df_active["Product_ID"] == sel_prod_id].iloc[0]

        curr_stock = float(prod_row["Current_Stock"])
        unit_str = str(prod_row.get("Unit", "units"))
        unit_price = float(prod_row["Selling_Price"])
        rec_disc = float(prod_row.get("Recommended_Discount", 0.0))

        st.markdown("<br>", unsafe_allow_html=True)
        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("Selected Product", prod_row["Product_Name"])
        bc2.metric("Available Stock", f"{curr_stock:.0f} {unit_str}")
        bc3.metric("Selling Price", f"{currency_sym}{unit_price:.2f}")
        bc4.metric("AI Recommended Discount", f"{rec_disc:.0f}%")

        st.markdown("---")
        with st.form("sell_product_form"):
            s_col1, s_col2 = st.columns(2)
            with s_col1:
                qty_sold = st.number_input(f"Quantity to Sell ({unit_str}) *", min_value=0.5, max_value=float(curr_stock + 100), value=min(5.0, curr_stock), step=1.0)
            with s_col2:
                disc_applied = st.number_input("Applied Discount (%)", min_value=0.0, max_value=100.0, value=rec_disc, step=1.0)

            # Bill preview calculation
            subtotal = qty_sold * unit_price
            disc_amt = subtotal * (disc_applied / 100.0)
            final_total = subtotal - disc_amt

            st.markdown(f"""
                <div class="receipt-card">
                    <h4>🧾 Sales Receipt Preview</h4>
                    <b>Product:</b> {prod_row['Product_Name']} ({prod_row['Product_ID']})<br>
                    <b>Quantity:</b> {qty_sold} {unit_str} × {currency_sym}{unit_price:.2f}<br>
                    <b>Subtotal:</b> {currency_sym}{subtotal:,.2f}<br>
                    <b>Discount ({disc_applied:.0f}%):</b> -{currency_sym}{disc_amt:,.2f}<br>
                    <hr>
                    <b style="font-size: 1.2rem; color: #10b981;">Final Total: {currency_sym}{final_total:,.2f}</b>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            submit_sale = st.form_submit_button("💳 Complete Sale & Print Bill", use_container_width=True)

        if submit_sale:
            if qty_sold > curr_stock:
                st.error(f"❌ Insufficient stock. Available quantity: {curr_stock:.0f} {unit_str}. You cannot sell {qty_sold} {unit_str}.")
            else:
                try:
                    df_new, sale_rec, msg = record_product_sale(df_active, sel_prod_id, qty_sold, disc_applied)
                    save_inventory_dataframe(df_new)
                    refresh_predictions()
                    st.success(f"✅ {msg}")
                    st.rerun()
                except Exception as ex:
                    st.error(f"❌ Sale failed: {ex}")

    # ----------------------------------------------------
    # TAB 5: TRANSACTION AUDIT LOGS
    # ----------------------------------------------------
    with inv_tab5:
        st.markdown("#### 📜 Inventory Transaction Audit Trail & Sales History")
        
        log_sub1, log_sub2 = st.tabs(["🔄 Inventory Movement Audit Log", "💳 Completed Sales History"])
        
        with log_sub1:
            df_trans_log = get_transaction_history()
            st.markdown(f"**Total Transactions Recorded: {len(df_trans_log)}**")
            st.dataframe(df_trans_log, use_container_width=True, height=400)
            
            st.download_button(
                label="📥 Download Inventory Audit Log CSV",
                data=df_trans_log.to_csv(index=False).encode('utf-8'),
                file_name="ecobite_inventory_transactions.csv",
                mime="text/csv"
            )

        with log_sub2:
            df_sales_log = get_sales_history()
            st.markdown(f"**Total Sales Bills Generated: {len(df_sales_log)}**")
            st.dataframe(df_sales_log, use_container_width=True, height=400)

            st.download_button(
                label="📥 Download Sales History CSV",
                data=df_sales_log.to_csv(index=False).encode('utf-8'),
                file_name="ecobite_sales_history.csv",
                mime="text/csv"
            )

    # ----------------------------------------------------
    # TAB 6: STOCK REFILL & LOW STOCK MANAGEMENT
    # ----------------------------------------------------
    with inv_tab6:
        st.markdown("#### 🔄 Stock Refill & Reorder Center")
        st.caption("Identify items with low or depleted stock, calculate safety stock & reorder points, and execute single or batch inventory refills.")

        # Compute refill metrics for active inventory
        df_refill = compute_refill_dataframe(df_active)

        # KPI Summary Row
        rf_tot_prods = len(df_refill)
        rf_low_stock = len(df_refill[df_refill["Stock_Status"] == "LOW STOCK"])
        rf_out_stock = len(df_refill[df_refill["Stock_Status"] == "OUT OF STOCK"])
        rf_reorder_req = len(df_refill[df_refill["Stock_Status"] == "REORDER REQUIRED"])
        rf_total_units = int(df_refill["Recommended_Refill"].sum())

        rk1, rk2, rk3, rk4, rk5 = st.columns(5)
        rk1.metric("📦 Total Products", f"{rf_tot_prods:,}")
        rk2.metric("⚠️ Low Stock SKUs", f"{rf_low_stock:,}")
        rk3.metric("🔴 Out of Stock SKUs", f"{rf_out_stock:,}", delta=f"{rf_out_stock/max(1,rf_tot_prods)*100:.1f}% of total" if rf_out_stock > 0 else "0%", delta_color="inverse")
        rk4.metric("🔄 Reorder Required SKUs", f"{rf_reorder_req:,}")
        rk5.metric("📈 Rec. Refill Units", f"{rf_total_units:,}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 🔍 Low Stock Search & Filters")

        f_col1, f_col2, f_col3, f_col4 = st.columns([1.5, 1.2, 1.2, 2])
        with f_col1:
            status_filter = st.selectbox(
                "Filter Stock Status",
                ["All Statuses", "Reorder Required & Low Stock", "Out of Stock", "Reorder Required", "Low Stock", "Healthy Stock"],
                index=1,
                key="rf_status_filter_sel"
            )
        with f_col2:
            cat_rf_opts = ["All Categories"] + sorted(list(df_refill["Category"].unique()))
            cat_rf_filter = st.selectbox("Filter Category", cat_rf_opts, index=0, key="rf_cat_filter_sel")
        with f_col3:
            urg_opts = ["All Urgency Levels", "REFILL IMMEDIATELY", "REFILL SOON", "PLAN REFILL", "NORMAL"]
            urg_filter = st.selectbox("Filter Urgency", urg_opts, index=0, key="rf_urg_filter_sel")
        with f_col4:
            rf_search = st.text_input("🔍 Search Product Name or ID", "", key="refill_search_q_input")

        # Apply Filters
        filtered_rf = df_refill.copy()

        if status_filter == "Reorder Required & Low Stock":
            filtered_rf = filtered_rf[filtered_rf["Stock_Status"].isin(["OUT OF STOCK", "REORDER REQUIRED", "LOW STOCK"])]
        elif status_filter == "Out of Stock":
            filtered_rf = filtered_rf[filtered_rf["Stock_Status"] == "OUT OF STOCK"]
        elif status_filter == "Reorder Required":
            filtered_rf = filtered_rf[filtered_rf["Stock_Status"] == "REORDER REQUIRED"]
        elif status_filter == "Low Stock":
            filtered_rf = filtered_rf[filtered_rf["Stock_Status"] == "LOW STOCK"]
        elif status_filter == "Healthy Stock":
            filtered_rf = filtered_rf[filtered_rf["Stock_Status"] == "HEALTHY STOCK"]

        if cat_rf_filter != "All Categories":
            filtered_rf = filtered_rf[filtered_rf["Category"] == cat_rf_filter]

        if urg_filter != "All Urgency Levels":
            filtered_rf = filtered_rf[filtered_rf["Urgency"] == urg_filter]

        if rf_search.strip():
            filtered_rf = fuzzy_search_dataframe(filtered_rf, rf_search, ["Product_Name", "Product_ID", "Category"])

        st.markdown(f"**Displaying {len(filtered_rf)} of {len(df_refill)} products requiring attention:**")

        rf_disp_cols = [
            "Product_ID", "Product_Name", "Category", "Current_Stock", "Unit",
            "Minimum_Stock", "Supplier_Lead_Time", "Reorder_Point", "Recommended_Refill",
            "Stock_Status", "Urgency", "Supplier"
        ]

        rf_table_show = filtered_rf[rf_disp_cols].copy()
        st.dataframe(rf_table_show, use_container_width=True, height=350)

        # Refill Action Section (Single & Batch)
        st.markdown("---")
        st.markdown("### 🔄 Execute Stock Refill")

        rf_sub_tab1, rf_sub_tab2 = st.tabs(["📦 Single Product Refill", "📋 Multi-Product Batch Refill"])

        with rf_sub_tab1:
            if not filtered_rf.empty:
                refill_options = filtered_rf["Product_ID"] + " - " + filtered_rf["Product_Name"] + " (Current: " + filtered_rf["Current_Stock"].astype(int).astype(str) + " " + filtered_rf["Unit"] + " | Rec Refill: " + filtered_rf["Recommended_Refill"].astype(int).astype(str) + ")"
                sel_rf_str = st.selectbox("Select Product to Refill:", refill_options, index=0, key="rf_single_product_sel")
                sel_rf_pid = sel_rf_str.split(" - ")[0]
                rf_prod_info = df_refill[df_refill["Product_ID"] == sel_rf_pid].iloc[0]

                st.markdown("<br>", unsafe_allow_html=True)
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Product Name", rf_prod_info["Product_Name"])
                rc2.metric("Current Stock", f"{rf_prod_info['Current_Stock']:.0f} {rf_prod_info['Unit']}")
                rc3.metric("Reorder Point Threshold", f"{rf_prod_info['Reorder_Point']:.0f} {rf_prod_info['Unit']}")
                rc4.metric("Recommended Refill", f"{rf_prod_info['Recommended_Refill']:.0f} {rf_prod_info['Unit']}")

                with st.form("single_refill_form"):
                    refill_input_qty = st.number_input(
                        f"Confirm Refill Quantity ({rf_prod_info['Unit']}) *",
                        min_value=1.0,
                        value=float(max(1.0, rf_prod_info['Recommended_Refill'])),
                        step=5.0
                    )
                    refill_notes = st.text_input("Refill Notes (Optional)", value=f"Refill from {rf_prod_info['Supplier']}")

                    submit_refill = st.form_submit_button("✅ Confirm & Execute Refill", use_container_width=True)

                if submit_refill:
                    if refill_input_qty <= 0:
                        st.error("❌ Refill quantity must be greater than zero.")
                    else:
                        try:
                            df_new, trans_rec, msg = refill_product_stock(df_active, sel_rf_pid, refill_input_qty, notes=refill_notes)
                            save_inventory_dataframe(df_new)
                            refresh_predictions()
                            st.success(f"✅ {msg}")
                            st.rerun()
                        except Exception as ex:
                            st.error(f"❌ Error performing refill: {ex}")
            else:
                st.info("No products match the selected refill filters.")

        with rf_sub_tab2:
            st.markdown("##### 📋 Batch Selection for Multi-Product Refill")
            low_stock_candidates = df_refill[df_refill["Stock_Status"].isin(["OUT OF STOCK", "REORDER REQUIRED", "LOW STOCK"])].copy()
            
            if not low_stock_candidates.empty:
                candidate_opts = low_stock_candidates["Product_ID"] + " - " + low_stock_candidates["Product_Name"] + " (Rec: " + low_stock_candidates["Recommended_Refill"].astype(int).astype(str) + " " + low_stock_candidates["Unit"] + ")"
                selected_batch_pids = st.multiselect(
                    "Select Low-Stock Products to Refill in Batch:",
                    candidate_opts,
                    default=candidate_opts[:min(5, len(candidate_opts))],
                    key="batch_refill_pids_sel"
                )

                if selected_batch_pids:
                    st.markdown("##### 📄 Selected Batch Refill Summary:")
                    batch_refill_items = []
                    for b_str in selected_batch_pids:
                        b_pid = b_str.split(" - ")[0]
                        b_row = df_refill[df_refill["Product_ID"] == b_pid].iloc[0]
                        batch_refill_items.append({
                            "Product_ID": b_pid,
                            "Product_Name": b_row["Product_Name"],
                            "Current_Stock": b_row["Current_Stock"],
                            "Refill_Qty": b_row["Recommended_Refill"],
                            "Unit": b_row["Unit"],
                            "Supplier": b_row["Supplier"]
                        })
                    
                    df_batch_summary = pd.DataFrame(batch_refill_items)
                    st.dataframe(df_batch_summary, use_container_width=True)

                    if st.button("🚀 Confirm & Process Selected Batch Refills", use_container_width=True):
                        refill_payload = [{"product_id": item["Product_ID"], "refill_qty": item["Refill_Qty"]} for item in batch_refill_items]
                        df_new, success_cnt, errors = refill_batch_products(df_active, refill_payload)
                        save_inventory_dataframe(df_new)
                        refresh_predictions()
                        if errors:
                            st.warning(f"⚠️ Batch complete: {success_cnt} succeeded, {len(errors)} failed: {errors}")
                        else:
                            st.success(f"✅ Successfully refilled {success_cnt} products in batch!")
                        st.rerun()
            else:
                st.success("🟢 All products currently have healthy inventory levels!")

# ----------------------------------------------------
# PAGE 3: SINGLE PRODUCT DEEP-DIVE
# ----------------------------------------------------
elif nav_choice == "🔍 Single Product Deep-Dive":
    st.subheader("🔍 Single Product Inventory & Waste Analysis")

    selected_product_id = st.selectbox(
        "Select Product to Analyze:",
        df_active["Product_ID"] + " - " + df_active["Product_Name"],
        index=0
    )
    
    prod_id = selected_product_id.split(" - ")[0]
    p_data = df_active[df_active["Product_ID"] == prod_id].iloc[0]
    unit_str = str(p_data.get("Unit", "units"))

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown(f"### **{p_data['Product_Name']}**")
        st.markdown(f"**Category:** {p_data['Category']} | **Season:** {p_data['Season']}")
        st.markdown(f"**Risk Level:** {get_risk_badge_html(p_data['Risk_Level'])} | **Status:** {get_inventory_status_badge_html(p_data['Risk_Level'], p_data['Days_to_Expiry'], p_data['Current_Stock'], p_data['Excess_Inventory'])}", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Current Stock", f"{float(p_data['Current_Stock']):.0f} {unit_str}")
        m2.metric("Predicted Demand", f"{float(p_data['Predicted_Demand']):.0f} {unit_str}")
        m3.metric("Excess Stock", f"{float(p_data['Excess_Inventory']):.0f} {unit_str}", delta_color="inverse")

        m4, m5, m6 = st.columns(3)
        m4.metric("Days to Expiry", f"{int(p_data['Days_to_Expiry'])} days")
        m5.metric("Waste Probability", f"{p_data['Waste_Probability']*100:.1f}%")
        m6.metric("Selling Price", f"{currency_sym}{p_data['Selling_Price']:.2f}")

    with col_right:
        st.markdown("#### 💡 AI Recommendation & Explanation")
        st.markdown(f"""
            <div class="explanation-card">
                <b>Recommended Discount:</b> <span style="font-size: 1.4rem; color: #3b82f6; font-weight: bold;">{p_data['Recommended_Discount']:.0f}%</span><br>
                <b>Recommended Selling Price:</b> {currency_sym}{p_data['Recommended_Price']:.2f}<br>
                <b>Primary Action:</b> {p_data['Recommended_Action']}<br>
                <b>Reorder Decision:</b> {p_data['Reorder_Status']}<br><br>
                <b>Explainable AI Rationale:</b><br>
                {p_data['Explanation']}
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📊 Stock vs Forecast Comparison")
    comp_df = pd.DataFrame({
        "Metric": ["Current Inventory Stock", "Predicted Demand Before Expiry", "Excess Inventory"],
        "Units": [p_data["Current_Stock"], p_data["Predicted_Demand"], p_data["Excess_Inventory"]]
    })
    fig_comp = px.bar(comp_df, x="Metric", y="Units", color="Metric", color_discrete_sequence=["#ef4444", "#10b981", "#f59e0b"])
    fig_comp.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig_comp, use_container_width=True)

# ----------------------------------------------------
# PAGE 4: INVENTORY ACTION CENTER
# ----------------------------------------------------
elif nav_choice == "📋 Inventory Action Center":
    st.subheader("📋 Inventory Decision & Action Center")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        risk_filter = st.multiselect("Filter by Risk Level", ["LOW", "MEDIUM", "HIGH", "CRITICAL"], default=["HIGH", "CRITICAL"])
    with col_f2:
        cat_filter = st.multiselect("Filter by Category", df_active["Category"].unique(), default=df_active["Category"].unique())
    with col_f3:
        search_query = st.text_input("Search Product Name or ID", "")

    # Apply filters
    filtered_df = df_active[
        (df_active["Risk_Level"].isin(risk_filter)) &
        (df_active["Category"].isin(cat_filter))
    ]
    if search_query:
        filtered_df = fuzzy_search_dataframe(filtered_df, search_query, ["Product_Name", "Product_ID", "Category"])

    st.markdown(f"**Displaying {len(filtered_df)} products requiring attention:**")

    display_cols = [
        "Product_ID", "Product_Name", "Category", "Current_Stock", "Unit", "Predicted_Demand",
        "Excess_Inventory", "Days_to_Expiry", "Waste_Probability", "Risk_Level",
        "Recommended_Discount", "Recommended_Price", "Recommended_Action", "Reorder_Status"
    ]
    
    st.dataframe(
        filtered_df[display_cols].style.format({
            "Waste_Probability": "{:.1%}",
            "Recommended_Discount": "{:.0f}%",
            "Recommended_Price": f"{currency_sym}" + "{:.2f}"
        }),
        use_container_width=True,
        height=400
    )

    # Export CSV button
    csv_buffer = filtered_df[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Action Plan CSV",
        data=csv_buffer,
        file_name="ecobite_inventory_action_plan.csv",
        mime="text/csv"
    )

# ----------------------------------------------------
# PAGE 5: MODEL PERFORMANCE & ML ANALYTICS
# ----------------------------------------------------
elif nav_choice == "📈 Model Performance & ML Analytics":
    st.subheader("📈 Machine Learning Performance & Explainability")

    import joblib
    meta_path = os.path.join(MODELS_DIR, "waste_model_meta.pkl")
    demand_meta_path = os.path.join(MODELS_DIR, "demand_model_meta.pkl")

    if os.path.exists(meta_path):
        meta_data = joblib.load(meta_path)
        rf_metrics = meta_data["rf_metrics"]
        comp_df = pd.DataFrame(meta_data["comparison"])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🎯 Waste Risk Classifier Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{rf_metrics['Accuracy']*100:.1f}%")
            m2.metric("Precision", f"{rf_metrics['Precision']*100:.1f}%")
            m3.metric("Recall (Priority)", f"{rf_metrics['Recall']*100:.1f}%")
            m4.metric("F1-Score", f"{rf_metrics['F1-Score']*100:.1f}%")

            st.markdown("##### Model Comparison Baseline")
            st.dataframe(comp_df, use_container_width=True)

        with c2:
            st.markdown("#### 🔲 Confusion Matrix (Waste Model)")
            cm = np.array(rf_metrics["Confusion_Matrix"])
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Predicted Label", y="Actual Label", color="Count"),
                x=['No Waste (0)', 'Waste (1)'],
                y=['No Waste (0)', 'Waste (1)'],
                color_continuous_scale="Blues"
            )
            fig_cm.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("---")
        st.markdown("#### ⭐ Random Forest Feature Importance")
        rf_model = predictor.waste_model
        feature_names = predictor.preprocessor.get_feature_names_out()
        importances = rf_model.feature_importances_
        
        feat_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(10)
        
        fig_feat = px.bar(feat_df, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Viridis")
        fig_feat.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_feat, use_container_width=True)

    if os.path.exists(demand_meta_path):
        st.markdown("---")
        st.markdown("#### 📉 Demand Prediction Regression Metrics")
        reg_metrics = joblib.load(demand_meta_path)
        d1, d2, d3 = st.columns(3)
        d1.metric("MAE (Mean Absolute Error)", f"{reg_metrics['MAE']:.2f} units")
        d2.metric("RMSE (Root Mean Sq Error)", f"{reg_metrics['RMSE']:.2f} units")
        d3.metric("R² Score", f"{reg_metrics['R2']:.4f}")

# ----------------------------------------------------
# PAGE 6: WHAT-IF DISCOUNT SIMULATOR (SECONDARY TOOL)
# ----------------------------------------------------
elif nav_choice == "⚙️ What-If Analysis Simulator":
    st.subheader("⚙️ Interactive What-If Discount & Scenario Simulator")
    st.caption("Adjust scenario inputs dynamically to test discount logic and observe revenue vs waste trade-offs.")

    col_sim_in, col_sim_out = st.columns([1, 1.2])

    with col_sim_in:
        st.markdown("#### 🎛️ Simulation Parameters")
        sim_name = st.text_input("Product Name", value="Sunscreen Lotion SPF 50")
        sim_stock = st.slider("Current Stock Units", min_value=10, max_value=500, value=140, step=5)
        sim_days = st.slider("Days to Expiry", min_value=1, max_value=90, value=15, step=1)
        sim_daily_sales = st.slider("Historical Daily Sales Velocity", min_value=1.0, max_value=50.0, value=7.0, step=0.5)
        sim_price = st.number_input("Original Selling Price", value=500.0, step=10.0)
        sim_prev_disc = st.slider("Previous Discount %", min_value=0.0, max_value=30.0, value=10.0, step=5.0)
        sim_disc_sales = st.slider("Sales Velocity During Discount", min_value=1.0, max_value=50.0, value=9.0, step=0.5)
        sim_season = st.selectbox("Current Season", ["Summer", "Winter", "Monsoon", "Spring"], index=0)

    # Run inference on simulated input
    sim_input = {
        "Product_ID": "SIM-001",
        "Product_Name": sim_name,
        "Category": "Personal Care",
        "Unit": "units",
        "Current_Stock": sim_stock,
        "Historical_Sales": sim_daily_sales * 30,
        "Daily_Sales": sim_daily_sales,
        "Selling_Price": sim_price,
        "Previous_Discount": sim_prev_disc,
        "Sales_During_Discount": sim_disc_sales,
        "Season": sim_season,
        "Month": 6,
        "Days_to_Expiry": sim_days,
        "Supplier_Lead_Time": 5,
        "Promotion": 1
    }

    sim_res = predictor.predict_single(sim_input)

    with col_sim_out:
        st.markdown("#### 📈 Simulation Output")

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Demand", f"{sim_res['Predicted_Demand']:.0f} units")
        m2.metric("Excess Stock", f"{sim_res['Excess_Inventory']:.0f} units")
        m3.metric("Waste Probability", f"{sim_res['Waste_Probability']*100:.1f}%")

        st.markdown(f"""
            <div class="explanation-card">
                <span style="font-size: 1.2rem;">Risk Level: <b>{get_risk_badge_html(sim_res['Risk_Level'])}</b></span><br><br>
                <b>Recommended Discount:</b> <span style="font-size: 1.8rem; color: #3b82f6; font-weight: bold;">{sim_res['Recommended_Discount']:.0f}%</span><br>
                <b>Original Price:</b> {currency_sym}{sim_price:.2f} ➔ <b>New Price:</b> <span style="color: #10b981; font-weight: bold;">{currency_sym}{sim_res['Recommended_Price']:.2f}</span><br><br>
                <b>Recommended Action:</b> {sim_res['Recommended_Action']}<br>
                <b>Reorder Action:</b> {sim_res['Reorder_Status']}<br><br>
                <b>AI Rationale:</b><br>{sim_res['Explanation']}
            </div>
        """, unsafe_allow_html=True)

        disc_calc = calculate_dynamic_discount(
            waste_probability=sim_res["Waste_Probability"],
            current_stock=sim_stock,
            predicted_demand=sim_res["Predicted_Demand"],
            days_to_expiry=sim_days,
            daily_sales=sim_daily_sales,
            selling_price=sim_price,
            previous_discount=sim_prev_disc,
            sales_during_discount=sim_disc_sales
        )

        rev_df = pd.DataFrame({
            "Scenario": ["Expected Disc. Revenue", "Est. Waste Value Saved"],
            "Amount": [disc_calc["expected_discounted_revenue"], disc_calc["waste_saved_value"]]
        })
        fig_rev = px.bar(rev_df, x="Scenario", y="Amount", color="Scenario", color_discrete_sequence=["#3b82f6", "#10b981"])
        fig_rev.update_layout(template="plotly_dark", height=250)
        st.plotly_chart(fig_rev, use_container_width=True)
