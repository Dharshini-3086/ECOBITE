"""
Waste Risk Classification Model Training Script for ECOBITE
Trains Random Forest Classifier and baseline models to predict product waste risk (Waste_Label).
Evaluates models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
"""

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from src.data_preprocessing import build_preprocessor_pipeline, prepare_features_and_targets

def train_waste_classification_model(data_path: str = None, models_dir: str = None):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if data_path is None:
        data_path = os.path.join(project_root, "data", "raw", "supermarket_inventory.csv")
    if models_dir is None:
        models_dir = os.path.join(project_root, "models")

    os.makedirs(models_dir, exist_ok=True)

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Separate features and target
    X, y = prepare_features_and_targets(df, target_col="Waste_Label")

    # Stratified Train/Test split to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

    # Fit preprocessor strictly on training data (no target leakage)
    preprocessor = build_preprocessor_pipeline()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # 1. Primary Model: Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        class_weight='balanced'
    )
    rf_clf.fit(X_train_trans, y_train)
    y_pred_rf = rf_clf.predict(X_test_trans)
    y_proba_rf = rf_clf.predict_proba(X_test_trans)[:, 1]

    # Evaluation metrics
    rf_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_rf),
        "Precision": precision_score(y_test, y_pred_rf, zero_division=0),
        "Recall": recall_score(y_test, y_pred_rf, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred_rf, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba_rf),
        "Confusion_Matrix": confusion_matrix(y_test, y_pred_rf).tolist()
    }

    print("\n--- Primary Model (Random Forest) Results ---")
    for metric, val in rf_metrics.items():
        if metric != "Confusion_Matrix":
            print(f"{metric}: {val:.4f}")
        else:
            print(f"Confusion Matrix:\n{np.array(val)}")

    # 2. Baseline Model Comparison
    models_to_compare = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42, class_weight='balanced'),
        "Random Forest": rf_clf
    }

    comparison_results = {}
    for name, model in models_to_compare.items():
        if name != "Random Forest":
            model.fit(X_train_trans, y_train)
        preds = model.predict(X_test_trans)
        probas = model.predict_proba(X_test_trans)[:, 1] if hasattr(model, "predict_proba") else preds
        
        comparison_results[name] = {
            "Accuracy": round(accuracy_score(y_test, preds), 4),
            "Precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "Recall": round(recall_score(y_test, preds, zero_division=0), 4),
            "F1-Score": round(f1_score(y_test, preds, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y_test, probas), 4)
        }

    print("\n--- Model Comparison Summary ---")
    df_comp = pd.DataFrame(comparison_results).T
    print(df_comp)

    # Save artifacts
    waste_model_path = os.path.join(models_dir, "waste_model.pkl")
    preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
    meta_path = os.path.join(models_dir, "waste_model_meta.pkl")

    joblib.dump(rf_clf, waste_model_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    meta_data = {
        "rf_metrics": rf_metrics,
        "comparison": comparison_results,
        "feature_names": preprocessor.get_feature_names_out().tolist()
    }
    joblib.dump(meta_data, meta_path)

    print(f"\nModels successfully saved to:\n  - {waste_model_path}\n  - {preprocessor_path}")
    return rf_metrics, df_comp

if __name__ == "__main__":
    train_waste_classification_model()
