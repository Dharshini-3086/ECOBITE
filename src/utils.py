"""
Utility Helper Module for ECOBITE
Contains CSS injection styles, Plotly theme helpers, KPI card components, and badge formatters.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def apply_custom_css():
    st.markdown("""
        <style>
        /* Modern Eco-Theme & Retail Intelligence Dashboard */
        .main {
            background-color: #0b132b;
            color: #f1f5f9;
        }
        .stMetric {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
            border: 1px solid rgba(16, 185, 129, 0.2);
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        .stMetric:hover {
            border-color: #10b981;
            transform: translateY(-2px);
            transition: all 0.2s ease;
        }
        .badge-low {
            background-color: #059669;
            color: #ffffff;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .badge-medium {
            background-color: #d97706;
            color: #ffffff;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .badge-high {
            background-color: #dc2626;
            color: #ffffff;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .badge-critical {
            background-color: #991b1b;
            color: #ffffff;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.85rem;
            border: 1px solid #ef4444;
        }
        .badge-healthy {
            background-color: #047857;
            color: #ffffff;
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: 600;
        }
        .badge-monitor {
            background-color: #b45309;
            color: #ffffff;
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: 600;
        }
        .badge-action {
            background-color: #c2410c;
            color: #ffffff;
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: 600;
        }
        .badge-urgent {
            background-color: #b91c1c;
            color: #ffffff;
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: 700;
        }
        .explanation-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.98));
            border-left: 5px solid #10b981;
            padding: 18px;
            border-radius: 10px;
            margin: 12px 0;
            font-size: 1.05rem;
            line-height: 1.6;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        }
        .receipt-card {
            background: #1e293b;
            border: 2px dashed #10b981;
            border-radius: 12px;
            padding: 20px;
            margin-top: 15px;
            font-family: 'Courier New', Courier, monospace;
        }
        .step-header {
            color: #10b981;
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 8px;
            border-bottom: 2px solid #10b981;
            padding-bottom: 4px;
        }
        </style>
    """, unsafe_allow_html=True)

def validate_uploaded_csv(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Validates uploaded CSV structure for required columns.
    """
    required_cols = [
        "Product_ID", "Product_Name", "Category", "Current_Stock",
        "Historical_Sales", "Daily_Sales", "Selling_Price", "Season",
        "Days_to_Expiry"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return False, missing
    return True, []

def get_risk_badge_html(risk_level: str) -> str:
    r = str(risk_level).upper()
    if r == "LOW":
        return '<span class="badge-low">🟢 LOW RISK</span>'
    elif r == "MEDIUM":
        return '<span class="badge-medium">🟡 MEDIUM RISK</span>'
    elif r == "HIGH":
        return '<span class="badge-high">🟠 HIGH RISK</span>'
    else:
        return '<span class="badge-critical">🔴 CRITICAL RISK</span>'

def get_inventory_status_badge_html(risk_level: str, days_to_expiry: float, stock: float, excess: float) -> str:
    if risk_level == "CRITICAL" or days_to_expiry <= 3:
        return '<span class="badge-urgent">🔴 Urgent Action</span>'
    elif risk_level == "HIGH" or excess > 0:
        return '<span class="badge-action">🟠 Action Required</span>'
    elif risk_level == "MEDIUM":
        return '<span class="badge-monitor">🟡 Monitor</span>'
    else:
        return '<span class="badge-healthy">🟢 Healthy</span>'

import re
from difflib import SequenceMatcher

def match_query_string(target_text: str, query_str: str) -> bool:
    if not query_str or pd.isna(target_text):
        return True
    
    target_clean = str(target_text).strip().lower()
    query_clean = str(query_str).strip().lower()
    
    if query_clean in target_clean:
        return True
        
    target_words = re.findall(r'\w+', target_clean)
    query_words = re.findall(r'\w+', query_clean)
    
    for q_w in query_words:
        if len(q_w) < 2:
            continue
        for t_w in target_words:
            if q_w in t_w or t_w in q_w:
                return True
            ratio = SequenceMatcher(None, q_w, t_w).ratio()
            if ratio >= 0.60 or (len(q_w) >= 3 and q_w[:2] == t_w[:2] and ratio >= 0.50):
                return True
            t_cons = re.sub(r'[aeiou]', '', t_w)
            q_cons = re.sub(r'[aeiou]', '', q_w)
            if len(q_cons) >= 2 and (q_cons in t_cons or t_cons in q_cons):
                return True
                
    return False

def fuzzy_search_dataframe(df: pd.DataFrame, search_query: str, search_cols: list) -> pd.DataFrame:
    if not search_query or not str(search_query).strip():
        return df
    
    q = str(search_query).strip()
    
    def row_matches(row):
        for col in search_cols:
            if col in row and match_query_string(row[col], q):
                return True
        return False
        
    mask = df.apply(row_matches, axis=1)
    return df[mask]

def get_stock_status_badge_html(status_str: str) -> str:
    s = str(status_str).upper()
    if "HEALTHY" in s:
        return '<span class="badge-healthy">🟢 HEALTHY STOCK</span>'
    elif "LOW" in s:
        return '<span class="badge-monitor">🟡 LOW STOCK</span>'
    elif "REORDER" in s:
        return '<span class="badge-action">🟠 REORDER REQUIRED</span>'
    else:
        return '<span class="badge-urgent">🔴 OUT OF STOCK</span>'

def get_urgency_badge_html(urgency_str: str) -> str:
    u = str(urgency_str).upper()
    if "IMMEDIATELY" in u or "CRITICAL" in u:
        return '<span class="badge-critical">🔴 REFILL IMMEDIATELY</span>'
    elif "SOON" in u:
        return '<span class="badge-high">🟠 REFILL SOON</span>'
    elif "PLAN" in u:
        return '<span class="badge-medium">🟡 PLAN REFILL</span>'
    else:
        return '<span class="badge-low">🟢 NORMAL</span>'


