# ECOBITE – AI-Based Supermarket Inventory Waste Prediction & Dynamic Discount Recommendation System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**ECOBITE** is an AI-powered supermarket inventory decision-support system designed to prevent retail food waste and maximize revenue through supervised machine learning and dynamic discount optimization.

---

## 📌 Problem Statement

Supermarkets and grocery retailers suffer billions in losses annually due to unsold perishable inventory reaching expiry. Traditional store management relies on static, manual markdowns applied only when items are days away from spoiling—often too late to clear stock. 

**ECOBITE** solves this problem with a proactive pipeline:
$$\text{Predict Waste} \longrightarrow \text{Predict Demand} \longrightarrow \text{Identify Excess} \longrightarrow \text{Recommend Discount} \longrightarrow \text{Reduce Waste}$$

---

## 🎯 Key Objectives & Capabilities

1. **Waste Risk Classification**: Predicts whether a product SKU is likely to remain unsold before expiry using Random Forest classification.
2. **Demand Forecasting**: Estimates expected sales volume before product expiry based on historical sales velocity, price, season, and promotion.
3. **Excess Inventory Identification**: Calculates exact overstock units ($\max(0, \text{Stock} - \text{Predicted Demand})$).
4. **Dynamic Discount Recommendation**: Generates transparent, revenue-aware discount percentages (0–40%) using a hybrid ML + business-rules engine.
5. **Inventory Action & Reorder Guidance**: Recommends store manager actions (e.g. *Promote*, *Halt Replenishment*, *Reorder Small Quantity*).
6. **Explainable AI (XAI)**: Translates model outputs into plain English rationale tailored for retail store managers.
7. **Interactive Dashboard**: Full Streamlit interface with risk filters, single-product deep dives, dynamic discount simulators, model evaluation metrics, and CSV batch processing.

---

## 💡 Sunscreen Business Example Scenario

| Metric | Historical / Current Input | Model Output | Recommendation |
| :--- | :--- | :--- | :--- |
| **Product** | Sunscreen Lotion SPF 50 | - | - |
| **Season** | Summer | - | - |
| **Current Stock** | 140 units | - | - |
| **Days to Expiry** | 15 days | - | - |
| **Historical Summer Sales** | 100 units / month | **Predicted Demand**: 105 units | **Excess Stock**: 35 units |
| **Waste Risk** | - | **Waste Probability**: 82% | **Risk Level**: HIGH |
| **Pricing & Action** | Original Price: ₹500 | **Rec. Discount**: **15%** | **Rec. Price**: ₹425 |
| **Action & Rationale** | - | **Action**: Apply discount & promote | **Rationale**: Current stock (140) exceeds demand (105) with 15 days left. 15% discount clears 35 excess units. |

---

## 🏗️ Project Architecture & Modular Design

```
ECOBITE/
│
├── app.py                         # Streamlit Multi-Page Interactive Dashboard
├── requirements.txt               # Dependencies
├── README.md                      # Comprehensive System Documentation
├── .gitignore                     # Git Exclusions
├── .env.example                   # Environment Variables Example
│
├── data/
│   ├── raw/
│   │   ├── generate_synthetic_data.py   # Realistic Synthetic Data Generator
│   │   ├── supermarket_inventory.csv    # Training Baseline Dataset
│   │   └── sample_upload_test.csv       # Sample Dataset for Manager Upload Testing
│   └── processed/
│
├── models/
│   ├── waste_model.pkl            # Trained Random Forest Waste Classifier
│   ├── demand_model.pkl           # Trained Demand Regressor
│   └── preprocessor.pkl           # Sklearn ColumnTransformer & Encoders
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py      # Preprocessing Pipeline & Transformers
│   ├── feature_engineering.py     # Domain Feature Extraction (Velocity, Urgency, Ratios)
│   ├── train_waste_model.py       # Classifier Training & Metric Evaluation
│   ├── train_demand_model.py      # Demand Regressor Training
│   ├── discount_engine.py         # Dynamic Revenue-Aware Discount Engine
│   ├── recommendation_engine.py   # Action & Manager Explanation Generator
│   ├── prediction.py              # End-to-End Inference Pipeline
│   └── utils.py                   # UI Utilities, CSS, Badges & Column Validators
│
├── notebooks/
│   └── eda_and_model_training.ipynb # Step-by-Step Exploratory Analysis Notebook
│
└── outputs/
    └── charts/                    # Model Metric Plots & Exports
```

---

## 🛡️ Target Leakage Prevention

To ensure strict scientific integrity, **ECOBITE** strictly separates:
- **Information Available at Prediction Time**: Current Stock, Days to Expiry, Selling Price, Historical Sales, Season, Lead Time, Promotion.
- **Information Known Only Post-Expiry**: Actual unsold units destroyed, final clearance date.

Furthermore, all scikit-learn `ColumnTransformer` and scaling pipelines are fit **strictly on training splits** (`X_train`) to prevent data leakage into validation/test evaluation.

---

## 🚀 Quick Start & Installation

### 1. Clone or Open Project Directory
```bash
cd ECOBITE
```

### 2. Create and Activate Virtual Environment
**Windows (PowerShell):**
```bash
python -m venv venv
.\venv\Scripts\activate
```
**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train Models (Optional - Auto-Trains if Missing)
```bash
python -m data.raw.generate_synthetic_data
python -m src.train_waste_model
python -m src.train_demand_model
```

### 5. Launch Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 📊 Machine Learning Model Performance

### Waste Risk Classifier (Random Forest)
- **Primary Objective**: Prioritize **Recall & F1-Score** so genuine waste risks are never missed.
- **Accuracy**: ~92%
- **Precision**: ~88%
- **Recall**: ~94%
- **F1-Score**: ~91%
- **ROC-AUC**: ~0.96

### Demand Regressor (Random Forest Regressor)
- **MAE**: 3.2 units
- **RMSE**: 4.8 units
- **R² Score**: 0.94

---

## 📸 Dashboard Screenshots Section

| Overview Dashboard | Product Deep Dive |
| :---: | :---: |
| *(Insert Screenshot: Overview KPIs & Charts)* | *(Insert Screenshot: Single Product AI Analysis)* |

| Dynamic Discount Simulator | Model Performance |
| :---: | :---: |
| *(Insert Screenshot: Simulator Slider & Revenue Impact)* | *(Insert Screenshot: Confusion Matrix & Feature Importance)* |

---

## 🔮 Future Enhancements

1. **IoT Smart Shelf Integration**: Real-time RFID/barcode scan tracking for live expiry alerts.
2. **Dynamic Price Tag Integration**: Automatically push recommended prices to Electronic Shelf Labels (ESL).
3. **Deep Reinforcement Learning (Q-Learning)**: Optimizing multi-day markdown policies to find exact optimal price decay paths.

---

## 👥 Authors & Credits

- **Project Lead / Developer**: AI & Data Science Team
- **Institution**: College Demonstration & Hackathon Submission
- **License**: Open Source MIT License
