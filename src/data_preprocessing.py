"""
Data Preprocessing Pipeline Module for ECOBITE
Constructs scikit-learn ColumnTransformers and preprocessing pipelines to handle imputations, encodings, and scaling without target leakage.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.feature_engineering import get_feature_columns, add_engineered_features

def build_preprocessor_pipeline():
    """
    Builds an un-fit scikit-learn ColumnTransformer pipeline.
    """
    num_cols, cat_cols = get_feature_columns()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ],
        remainder='drop'
    )

    return preprocessor

def prepare_features_and_targets(df: pd.DataFrame, target_col: str = "Waste_Label"):
    """
    Applies feature engineering and separates features X and target y.
    """
    df_feat = add_engineered_features(df)
    num_cols, cat_cols = get_feature_columns()
    
    feature_cols = num_cols + cat_cols
    X = df_feat[feature_cols]
    
    y = df_feat[target_col] if target_col in df_feat.columns else None
    
    return X, y
