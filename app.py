import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ECOBITE",
    page_icon="🥬",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

DATA_FILE = "supermarket_inventory.csv"

FEATURES = [
    "category",
    "stock_quantity",
    "daily_sales",
    "days_to_expiry",
    "discount",
    "purchase_frequency",
    "price",
    "storage_method"
]

CATEGORICAL_FEATURES = [
    "category",
    "storage_method"
]

NUMERICAL_FEATURES = [
    "stock_quantity",
    "daily_sales",
    "days_to_expiry",
    "discount",
    "purchase_frequency",
    "price"
]


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_FILE)

    numerical_columns = [
        "stock_quantity",
        "daily_sales",
        "days_to_expiry",
        "discount",
        "purchase_frequency",
        "price",
        "waste"
    ]

    for column in numerical_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df = df.dropna(
        subset=FEATURES + ["waste"]
    ).reset_index(drop=True)

    return df


# ============================================================
# TRAIN MACHINE LEARNING MODELS
# ============================================================

@st.cache_resource
def train_models():

    df = load_data()

    X = df[FEATURES]
    y = df["waste"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # PREPROCESSOR
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES
            ),

            (
                "numerical",
                "passthrough",
                NUMERICAL_FEATURES
            )
        ]
    )

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=8,
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                random_state=42
            )
    }

    trained_models = {}

    results = []

    for model_name, model in models.items():

        pipeline = Pipeline(

            steps=[

                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "model",
                    model
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        trained_models[
            model_name
        ] = pipeline

        results.append({

            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1

        })

    results_df = pd.DataFrame(
        results
    )

    best_model_name = (

        results_df
        .sort_values(
            "F1 Score",
            ascending=False
        )
        .iloc[0]["Model"]

    )

    return (
        trained_models,
        results_df,
        best_model_name,
        X_test,
        y_test
    )


# ============================================================
# LOAD DATA + TRAIN MODELS
# ============================================================

try:

    df = load_data()

    (
        models,
        model_results,
        best_model_name,
        X_test,
        y_test

    ) = train_models()

    best_model = models[
        best_model_name
    ]

    random_forest_model = models[
        "Random Forest"
    ]

except FileNotFoundError:

    st.error(
        "supermarket_inventory.csv was not found."
    )

    st.info(
        "Make sure supermarket_inventory.csv "
        "is in the same folder as app.py."
    )

    st.stop()

except Exception as e:

    st.error(
        f"Application error: {e}"
    )

    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_risk_level(score):

    if score >= 70:

        return "High"

    elif score >= 40:

        return "Medium"

    else:

        return "Low"


# ------------------------------------------------------------
# EXPLANATION
# ------------------------------------------------------------

def get_explanation(
    stock,
    daily_sales,
    days_expiry,
    discount
):

    explanations = []

    expected_sales = (
        daily_sales *
        days_expiry
    )

    if days_expiry <= 3:

        explanations.append(
            "The product is very close to expiry."
        )

    elif days_expiry <= 7:

        explanations.append(
            "The product has limited time remaining "
            "before expiry."
        )

    if stock > expected_sales:

        explanations.append(
            "Current stock is higher than expected "
            "sales before expiry."
        )

    if daily_sales <= 3:

        explanations.append(
            "The current sales rate is relatively low."
        )

    if discount == 0 and days_expiry <= 7:

        explanations.append(
            "No discount is currently applied to a "
            "potentially risky product."
        )

    if not explanations:

        explanations.append(
            "The inventory and sales conditions "
            "indicate relatively low waste risk."
        )

    return explanations


# ------------------------------------------------------------
# SMART RECOMMENDATIONS
# ------------------------------------------------------------

def get_recommendation(
    stock,
    daily_sales,
    days_expiry,
    discount
):

    recommendations = []

    expected_sales = (
        daily_sales *
        days_expiry
    )

    excess_stock = max(
        0,
        stock - expected_sales
    )

    if days_expiry <= 3:

        recommendations.append(
            "Prioritize this product immediately "
            "because it is close to expiry."
        )

    if excess_stock > 0:

        recommendations.append(
            f"Reduce excess stock of approximately "
            f"{int(excess_stock)} units."
        )

    if discount == 0 and days_expiry <= 7:

        recommendations.append(
            "Consider applying a discount or promotion "
            "to increase sales before expiry."
        )

    if daily_sales <= 3:

        recommendations.append(
            "Sales velocity is low. Avoid purchasing "
            "large quantities in the next cycle."
        )

    if not recommendations:

        recommendations.append(
            "Continue monitoring sales and expiry."
        )

    return recommendations


# ------------------------------------------------------------
# PRIORITY LEVEL
# ------------------------------------------------------------

def get_priority_level(score):

    if score >= 70:

        return "🔴 Critical"

    elif score >= 40:

        return "🟠 High"

    elif score >= 20:

        return "🟡 Medium"

    else:

        return "🟢 Low"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🥬 ECOBITE"
)

st.sidebar.write(
    "Supermarket Inventory Intelligence"
)

st.sidebar.divider()

page = st.sidebar.radio(

    "Navigation",

    [
        "Dashboard",
        "Model Comparison",
        "Waste Prediction",
        "Daily Priority List",
        "Inventory Management"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "ECOBITE | AI & Data Science Capstone Project"
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title(
        "🥬 ECOBITE"
    )

    st.subheader(
        "AI-Based Household Food Waste Prediction "
        "and Intelligent Recommendation System"
    )

    st.write(
        "ECOBITE uses supervised machine learning "
        "to predict which grocery items are likely "
        "to be wasted before expiry."
    )

    st.divider()

    # --------------------------------------------------------
    # PREDICTIONS FOR ALL PRODUCTS
    # --------------------------------------------------------

    predictions = best_model.predict_proba(
        df[FEATURES]
    )[:, 1]

    risk_scores = (
        predictions * 100
    )

    high_risk = np.sum(
        risk_scores >= 70
    )

    medium_risk = np.sum(
        (risk_scores >= 40)
        &
        (risk_scores < 70)
    )

    low_risk = np.sum(
        risk_scores < 40
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Products",
        len(df)
    )

    col2.metric(
        "Total Stock",
        int(
            df["stock_quantity"].sum()
        )
    )

    col3.metric(
        "High Waste Risk",
        int(high_risk)
    )

    col4.metric(
        "Low Waste Risk",
        int(low_risk)
    )

    st.divider()

    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "📊 Waste Risk Distribution"
    )

    risk_distribution = pd.DataFrame({

        "Risk Level": [
            "High Risk",
            "Medium Risk",
            "Low Risk"
        ],

        "Number of Products": [
            high_risk,
            medium_risk,
            low_risk
        ]

    })

    st.bar_chart(
        risk_distribution.set_index(
            "Risk Level"
        )
    )

    st.divider()

    # --------------------------------------------------------
    # TOP RISK PRODUCTS
    # --------------------------------------------------------

    top_products = df[
        [
            "product",
            "category",
            "stock_quantity",
            "daily_sales",
            "days_to_expiry"
        ]
    ].copy()

    top_products[
        "waste_risk_score"
    ] = risk_scores

    top_products[
        "risk_level"
    ] = top_products[
        "waste_risk_score"
    ].apply(
        get_risk_level
    )

    top_products = (

        top_products
        .sort_values(
            "waste_risk_score",
            ascending=False
        )
        .head(10)

    )

    st.subheader(
        "🚨 Top 10 High-Risk Products"
    )

    st.dataframe(
        top_products,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        f"🏆 Best performing model: "
        f"**{best_model_name}**"
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    st.header(
        "🤖 Model Performance Comparison"
    )

    st.write(
        "ECOBITE compares three supervised machine "
        "learning algorithms."
    )

    st.divider()

    # --------------------------------------------------------
    # PERFORMANCE TABLE
    # --------------------------------------------------------

    display_results = (
        model_results.copy()
    )

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    for metric in metrics:

        display_results[metric] = (

            display_results[metric] * 100

        ).round(2)

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )

    st.success(
        f"🏆 Best Model: **{best_model_name}**"
    )

    st.divider()

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    st.subheader(
        "📊 Model Metrics"
    )

    chart_data = (
        model_results
        .set_index("Model")[metrics]
    )

    st.bar_chart(
        chart_data
    )

    st.divider()

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader(
        "🌲 Random Forest Confusion Matrix"
    )

    rf_predictions = (
        random_forest_model.predict(
            X_test
        )
    )

    cm = confusion_matrix(
        y_test,
        rf_predictions
    )

    confusion_df = pd.DataFrame(

        cm,

        index=[
            "Actual Low Risk",
            "Actual High Risk"
        ],

        columns=[
            "Predicted Low Risk",
            "Predicted High Risk"
        ]

    )

    st.dataframe(
        confusion_df,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader(
        "🌟 Random Forest Feature Importance"
    )

    try:

        preprocessor = (
            random_forest_model
            .named_steps[
                "preprocessor"
            ]
        )

        rf_classifier = (
            random_forest_model
            .named_steps[
                "model"
            ]
        )

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        importance_df = pd.DataFrame({

            "Feature":
                feature_names,

            "Importance":
                rf_classifier
                .feature_importances_

        })

        importance_df = (

            importance_df
            .sort_values(
                "Importance",
                ascending=False
            )
            .head(15)

        )

        st.bar_chart(
            importance_df.set_index(
                "Feature"
            )
        )

    except Exception:

        st.info(
            "Feature importance could not be displayed."
        )


# ============================================================
# WASTE PREDICTION
# ============================================================

elif page == "Waste Prediction":

    st.header(
        "🔍 Food Waste Prediction"
    )

    st.write(
        "Enter grocery item details and ECOBITE "
        "will predict its waste risk."
    )

    st.divider()

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    with col1:

        product_name = st.text_input(
            "Product Name",
            "Milk"
        )

        category = st.selectbox(
            "Category",
            sorted(
                df["category"]
                .dropna()
                .unique()
            )
        )

        stock_quantity = st.number_input(
            "Stock Quantity",
            min_value=0,
            value=100
        )

        daily_sales = st.number_input(
            "Average Daily Sales",
            min_value=0,
            value=5
        )

        days_to_expiry = st.number_input(
            "Days Until Expiry",
            min_value=1,
            value=2
        )

    with col2:

        discount = st.number_input(
            "Discount (%)",
            min_value=0,
            max_value=100,
            value=0
        )

        purchase_frequency = st.number_input(
            "Purchase Frequency (days)",
            min_value=1,
            value=7
        )

        price = st.number_input(
            "Product Price (₹)",
            min_value=0.0,
            value=60.0
        )

        storage_method = st.selectbox(
            "Storage Method",
            sorted(
                df["storage_method"]
                .dropna()
                .unique()
            )
        )

    st.write("")

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        "🚀 Predict Waste Risk",
        use_container_width=True
    ):

        input_data = pd.DataFrame({

            "category": [
                category
            ],

            "stock_quantity": [
                stock_quantity
            ],

            "daily_sales": [
                daily_sales
            ],

            "days_to_expiry": [
                days_to_expiry
            ],

            "discount": [
                discount
            ],

            "purchase_frequency": [
                purchase_frequency
            ],

            "price": [
                price
            ],

            "storage_method": [
                storage_method
            ]

        })

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = (
            best_model.predict(
                input_data
            )[0]
        )

        probability = (
            best_model.predict_proba(
                input_data
            )[0][1]
        )

        risk_score = (
            probability * 100
        )

        risk_level = get_risk_level(
            risk_score
        )

        st.divider()

        st.header(
            f"Prediction Result: {product_name}"
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        if prediction == 1:

            st.error(
                "⚠️ HIGH WASTE RISK"
            )

            st.write(
                "This product is likely to be "
                "wasted before expiry."
            )

        else:

            st.success(
                "✅ LOW WASTE RISK"
            )

            st.write(
                "This product is likely to be "
                "consumed before expiry."
            )

        # ----------------------------------------------------
        # RISK SCORE
        # ----------------------------------------------------

        st.subheader(
            "📈 Waste Risk Score"
        )

        st.progress(
            min(
                int(risk_score),
                100
            )
        )

        st.metric(
            "Waste Risk Score",
            f"{risk_score:.2f}%"
        )

        st.info(
            f"Risk Level: **{risk_level}**"
        )

        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        st.subheader(
            "💡 Prediction Explanation"
        )

        explanations = get_explanation(

            stock_quantity,
            daily_sales,
            days_to_expiry,
            discount

        )

        for explanation in explanations:

            st.write(
                f"• {explanation}"
            )

        # ----------------------------------------------------
        # RECOMMENDATIONS
        # ----------------------------------------------------

        st.subheader(
            "💡 Smart Recommendations"
        )

        recommendations = get_recommendation(

            stock_quantity,
            daily_sales,
            days_to_expiry,
            discount

        )

        for recommendation in recommendations:

            st.info(
                recommendation
            )

        # ----------------------------------------------------
        # PURCHASE SUGGESTION
        # ----------------------------------------------------

        st.subheader(
            "🛒 Smart Purchase Suggestion"
        )

        weekly_demand = int(
            daily_sales * 7
        )

        if stock_quantity > weekly_demand:

            excess = (
                stock_quantity -
                weekly_demand
            )

            st.warning(

                f"Current stock is "
                f"**{stock_quantity} units**, "
                f"while estimated weekly demand "
                f"is **{weekly_demand} units**. "
                f"Consider purchasing approximately "
                f"**{int(excess)} fewer units** "
                f"in the next cycle."

            )

        else:

            additional_purchase = (

                weekly_demand -
                stock_quantity

            )

            st.success(

                f"Estimated weekly demand: "
                f"**{weekly_demand} units**. "
                f"Suggested additional purchase: "
                f"**{additional_purchase} units**."

            )


# ============================================================
# DAILY PRIORITY LIST
# ============================================================

elif page == "Daily Priority List":

    st.header(
        "📋 Daily Priority List"
    )

    st.write(
        "Products are ranked according to expiry risk, "
        "excess stock and sales rate."
    )

    st.divider()

    priority_data = df.copy()

    # --------------------------------------------------------
    # EXPECTED SALES
    # --------------------------------------------------------

    priority_data[
        "expected_sales_before_expiry"
    ] = (

        priority_data[
            "daily_sales"
        ]
        *
        priority_data[
            "days_to_expiry"
        ]

    )

    # --------------------------------------------------------
    # EXCESS STOCK
    # --------------------------------------------------------

    priority_data[
        "excess_stock"
    ] = (

        priority_data[
            "stock_quantity"
        ]
        -
        priority_data[
            "expected_sales_before_expiry"
        ]

    ).clip(
        lower=0
    )

    # --------------------------------------------------------
    # PRIORITY SCORE
    # --------------------------------------------------------

    priority_data[
        "priority_score"
    ] = 0.0

    # Expiry contribution
    priority_data.loc[
        priority_data[
            "days_to_expiry"
        ] <= 3,
        "priority_score"
    ] += 50

    priority_data.loc[
        (
            priority_data[
                "days_to_expiry"
            ] > 3
        )
        &
        (
            priority_data[
                "days_to_expiry"
            ] <= 7
        ),
        "priority_score"
    ] += 30

    # Excess stock contribution
    stock_safe = (
        priority_data[
            "stock_quantity"
        ]
        .replace(
            0,
            1
        )
    )

    priority_data[
        "priority_score"
    ] += (

        priority_data[
            "excess_stock"
        ]
        /
        stock_safe

    ) * 30

    # Low sales contribution
    priority_data.loc[
        priority_data[
            "daily_sales"
        ] <= 3,
        "priority_score"
    ] += 20

    priority_data[
        "priority_score"
    ] = (

        priority_data[
            "priority_score"
        ]
        .clip(
            0,
            100
        )
        .round(2)

    )

    # --------------------------------------------------------
    # PRIORITY LEVEL
    # --------------------------------------------------------

    priority_data[
        "priority"
    ] = priority_data[
        "priority_score"
    ].apply(
        get_priority_level
    )

    priority_data = (

        priority_data
        .sort_values(
            "priority_score",
            ascending=False
        )

    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    critical = len(
        priority_data[
            priority_data[
                "priority_score"
            ] >= 70
        ]
    )

    high = len(
        priority_data[
            (
                priority_data[
                    "priority_score"
                ] >= 40
            )
            &
            (
                priority_data[
                    "priority_score"
                ] < 70
            )
        ]
    )

    medium = len(
        priority_data[
            (
                priority_data[
                    "priority_score"
                ] >= 20
            )
            &
            (
                priority_data[
                    "priority_score"
                ] < 40
            )
        ]
    )

    low = len(
        priority_data[
            priority_data[
                "priority_score"
            ] < 20
        ]
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "🔴 Critical",
        critical
    )

    col2.metric(
        "🟠 High",
        high
    )

    col3.metric(
        "🟡 Medium",
        medium
    )

    col4.metric(
        "🟢 Low",
        low
    )

    st.divider()

    # --------------------------------------------------------
    # PRIORITY TABLE
    # --------------------------------------------------------

    st.subheader(
        "🚨 Items Requiring Attention"
    )

    display_columns = [

        "product",
        "category",
        "stock_quantity",
        "daily_sales",
        "days_to_expiry",
        "excess_stock",
        "priority_score",
        "priority"

    ]

    st.dataframe(

        priority_data[
            display_columns
        ].head(30),

        use_container_width=True,

        hide_index=True

    )

    st.divider()

    # --------------------------------------------------------
    # TOP ACTIONS
    # --------------------------------------------------------

    st.subheader(
        "💡 Recommended Actions"
    )

    top_items = (
        priority_data
        .head(5)
    )

    for _, item in top_items.iterrows():

        st.markdown(
            f"### {item['product']}"
        )

        st.write(

            f"Priority Score: "
            f"**{item['priority_score']}** | "
            f"Stock: "
            f"**{int(item['stock_quantity'])}** | "
            f"Days to expiry: "
            f"**{int(item['days_to_expiry'])}**"

        )

        if item[
            "days_to_expiry"
        ] <= 3:

            st.error(
                "Action: Prioritize immediately. "
                "Consider a discount or promotion."
            )

        elif item[
            "excess_stock"
        ] > 0:

            st.warning(
                "Action: Reduce future purchasing "
                "and focus on selling existing stock."
            )

        else:

            st.info(
                "Action: Continue monitoring."
            )


# ============================================================
# INVENTORY MANAGEMENT
# ============================================================

elif page == "Inventory Management":

    st.header(
        "📦 Inventory Management"
    )

    st.write(
        "Add new products or update stock for "
        "existing products."
    )

    st.divider()

    # ========================================================
    # ADD / UPDATE INVENTORY
    # ========================================================

    st.subheader(
        "➕ Add / Update Inventory"
    )

    col1, col2 = st.columns(2)

    with col1:

        new_product = st.text_input(
            "Product Name",
            placeholder="Example: Milk"
        )

        categories = sorted(
            df[
                "category"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        new_category = st.selectbox(
            "Category",
            categories
        )

        new_stock = st.number_input(
            "Quantity to Add",
            min_value=1,
            value=20,
            step=1
        )

        new_daily_sales = st.number_input(
            "Average Daily Sales",
            min_value=0,
            value=5,
            step=1
        )

        new_days_expiry = st.number_input(
            "Days Until Expiry",
            min_value=1,
            value=7,
            step=1
        )

    with col2:

        new_discount = st.number_input(
            "Discount (%)",
            min_value=0,
            max_value=100,
            value=0,
            step=5
        )

        new_purchase_frequency = st.number_input(
            "Purchase Frequency (days)",
            min_value=1,
            value=7,
            step=1
        )

        new_price = st.number_input(
            "Product Price (₹)",
            min_value=0.0,
            value=50.0,
            step=1.0
        )

        storage_methods = sorted(
            df[
                "storage_method"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        new_storage = st.selectbox(
            "Storage Method",
            storage_methods
        )

    st.write("")

    # ========================================================
    # ADD / UPDATE BUTTON
    # ========================================================

    if st.button(
        "➕ Add / Update Product",
        use_container_width=True
    ):

        product_name_clean = (
            new_product.strip()
        )

        # ----------------------------------------------------
        # VALIDATE PRODUCT NAME
        # ----------------------------------------------------

        if product_name_clean == "":

            st.warning(
                "Please enter a product name."
            )

        else:

            # ------------------------------------------------
            # FIND EXISTING PRODUCT
            # ------------------------------------------------

            existing_index = df[
                df[
                    "product"
                ]
                .astype(str)
                .str.strip()
                .str.lower()
                ==
                product_name_clean.lower()
            ].index

            # =================================================
            # EXISTING PRODUCT
            # =================================================

            if len(existing_index) > 0:

                index = existing_index[0]

                # Current stock
                current_stock = int(
                    df.loc[
                        index,
                        "stock_quantity"
                    ]
                )

                # Additional stock
                additional_stock = int(
                    new_stock
                )

                # Updated stock
                updated_stock = (

                    current_stock
                    +
                    additional_stock

                )

                # ------------------------------------------------
                # UPDATE VALUES
                # ------------------------------------------------

                df.loc[
                    index,
                    "stock_quantity"
                ] = updated_stock

                df.loc[
                    index,
                    "category"
                ] = new_category

                df.loc[
                    index,
                    "daily_sales"
                ] = new_daily_sales

                df.loc[
                    index,
                    "days_to_expiry"
                ] = new_days_expiry

                df.loc[
                    index,
                    "discount"
                ] = new_discount

                df.loc[
                    index,
                    "purchase_frequency"
                ] = new_purchase_frequency

                df.loc[
                    index,
                    "price"
                ] = new_price

                df.loc[
                    index,
                    "storage_method"
                ] = new_storage

                # ------------------------------------------------
                # RECALCULATE WASTE LABEL
                # ------------------------------------------------

                expected_sales = (

                    new_daily_sales
                    *
                    new_days_expiry

                )

                waste_score = 0

                # Excess stock
                if updated_stock > expected_sales:

                    waste_score += 3

                # Expiry
                if new_days_expiry <= 3:

                    waste_score += 3

                elif new_days_expiry <= 7:

                    waste_score += 2

                elif new_days_expiry <= 14:

                    waste_score += 1

                # Low sales
                if new_daily_sales <= 3:

                    waste_score += 2

                elif new_daily_sales <= 7:

                    waste_score += 1

                # Frequent purchasing
                if new_purchase_frequency <= 5:

                    waste_score += 1

                # Discount
                if new_discount >= 15:

                    waste_score -= 1

                if waste_score >= 3:

                    updated_waste = 1

                else:

                    updated_waste = 0

                df.loc[
                    index,
                    "waste"
                ] = updated_waste

                # ------------------------------------------------
                # SAVE DATASET
                # ------------------------------------------------

                df.to_csv(
                    DATA_FILE,
                    index=False
                )

                # Clear cache
                load_data.clear()
                train_models.clear()

                st.success(
                    f"✅ {product_name_clean} "
                    "stock updated successfully!"
                )

                st.info(

                    f"Previous stock: "
                    f"**{current_stock} units**  \n"

                    f"Additional stock: "
                    f"**{additional_stock} units**  \n"

                    f"Updated stock: "
                    f"**{updated_stock} units**"

                )

                st.rerun()

            # =================================================
            # NEW PRODUCT
            # =================================================

            else:

                # ------------------------------------------------
                # CALCULATE WASTE LABEL
                # ------------------------------------------------

                expected_sales = (

                    new_daily_sales
                    *
                    new_days_expiry

                )

                waste_score = 0

                if new_stock > expected_sales:

                    waste_score += 3

                if new_days_expiry <= 3:

                    waste_score += 3

                elif new_days_expiry <= 7:

                    waste_score += 2

                elif new_days_expiry <= 14:

                    waste_score += 1

                if new_daily_sales <= 3:

                    waste_score += 2

                elif new_daily_sales <= 7:

                    waste_score += 1

                if new_purchase_frequency <= 5:

                    waste_score += 1

                if new_discount >= 15:

                    waste_score -= 1

                if waste_score >= 3:

                    new_waste = 1

                else:

                    new_waste = 0

                # ------------------------------------------------
                # CREATE NEW ROW
                # ------------------------------------------------

                new_row = pd.DataFrame({

                    "product": [
                        product_name_clean
                    ],

                    "category": [
                        new_category
                    ],

                    "stock_quantity": [
                        new_stock
                    ],

                    "daily_sales": [
                        new_daily_sales
                    ],

                    "days_to_expiry": [
                        new_days_expiry
                    ],

                    "discount": [
                        new_discount
                    ],

                    "purchase_frequency": [
                        new_purchase_frequency
                    ],

                    "price": [
                        new_price
                    ],

                    "storage_method": [
                        new_storage
                    ],

                    "waste": [
                        new_waste
                    ]

                })

                # ------------------------------------------------
                # ADD NEW PRODUCT
                # ------------------------------------------------

                updated_df = pd.concat(

                    [
                        df,
                        new_row
                    ],

                    ignore_index=True

                )

                updated_df.to_csv(
                    DATA_FILE,
                    index=False
                )

                # Clear cache
                load_data.clear()
                train_models.clear()

                st.success(
                    f"✅ {product_name_clean} "
                    "was successfully added!"
                )

                st.info(
                    f"Initial stock: "
                    f"**{new_stock} units**"
                )

                st.rerun()

    st.divider()

    # ========================================================
    # CURRENT INVENTORY
    # ========================================================

    st.subheader(
        "📋 Current Inventory"
    )

    current_inventory = load_data()

    st.dataframe(

        current_inventory,

        use_container_width=True,

        hide_index=True

    )

    st.divider()

    # ========================================================
    # SEARCH INVENTORY
    # ========================================================

    st.subheader(
        "🔎 Search Inventory"
    )

    search_product = st.text_input(
        "Search by product name"
    )

    if search_product:

        search_results = (

            current_inventory[
                current_inventory[
                    "product"
                ]
                .astype(str)
                .str.contains(
                    search_product,
                    case=False,
                    na=False
                )
            ]

        )

        if len(search_results) > 0:

            st.dataframe(

                search_results,

                use_container_width=True,

                hide_index=True

            )

        else:

            st.warning(
                "No matching products found."
            )

    st.divider()

    # ========================================================
    # PRODUCTS NEAR EXPIRY
    # ========================================================

    st.subheader(
        "⏰ Products Nearing Expiry"
    )

    near_expiry = (

        current_inventory[
            current_inventory[
                "days_to_expiry"
            ] <= 7
        ]

        .sort_values(
            "days_to_expiry"
        )

    )

    if len(near_expiry) > 0:

        st.dataframe(

            near_expiry,

            use_container_width=True,

            hide_index=True

        )

    else:

        st.success(
            "No products are currently "
            "near expiry."
        )

    st.divider()

    # ========================================================
    # HIGH STOCK PRODUCTS
    # ========================================================

    st.subheader(
        "📦 High Stock Products"
    )

    high_stock = (

        current_inventory[
            current_inventory[
                "stock_quantity"
            ] >= 100
        ]

        .sort_values(
            "stock_quantity",
            ascending=False
        )

    )

    if len(high_stock) > 0:

        st.dataframe(

            high_stock,

            use_container_width=True,

            hide_index=True

        )

    else:

        st.success(
            "No products currently have "
            "100+ units in stock."
        )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "ECOBITE — Predict • Explain • Prevent • Optimize"
)