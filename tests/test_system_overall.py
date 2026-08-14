"""
End-to-End System Test for Academic Performance & Behavioral Analytics Pipeline.
File: tests/test_system_overall.py

Verifies full system integration across both module suites:
[cbdatsi] Data Validation (22 cols) ---> Feature Engineering ---> K-Means Clustering ---> Chi-Square Inference
[cbadvai] Preprocessing (11 features) ---> 80/20 Stratified Split ---> 5-Fold CV ---> Multi-Model AI Performance Gates
"""

import os
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# ==============================================================================
# MODULE IMPORTS
# ==============================================================================
# Data Science & Analytics Suite
from src.cbdatsi.pipeline import (
    load_and_cache_dataset,
    validate_dataset,
    clean_and_typecast_data,
    perform_feature_engineering,
)
from src.cbdatsi.modeling import (
    run_kmeans_clustering,
    evaluate_clusters,
)
from src.cbdatsi.inference import perform_chisquare_independence

# Artificial Intelligence & ML Pipeline (cbadvai)
from src.cbadvai.preprocessing import (
    load_and_preprocess_data,
    get_train_test_split,
    SELECTED_FEATURES,
)
from src.cbadvai.models import (
    build_ordinal_lr_pipeline,
    build_mlp_pipeline,
    build_rf_pipeline
)
from src.cbadvai.metrics import evaluate_ordinal_model


# ==============================================================================
# MODEL-SPECIFIC PERFORMANCE GATE CONFIGURATION
# ==============================================================================
MODEL_THRESHOLDS = {
    "ordinal_lr": {
        "max_mae": 1.30,   
        "min_f1": 0.20,   
        "min_qwk": 0.05,
    },
    "mlp": {
        "max_mae": 1.12,  
        "min_f1": 0.20,    
        "min_qwk": 0.10,
    },
    "rf": {
        "max_mae": 0.70,   
        "min_f1": 0.24,   
        "min_qwk": 0.12,
    },
}


# ==============================================================================
# PIPELINE SETUP & GLOBAL STATE PREPARATION
# ==============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Database paper.xlsx")
CACHE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "dataset_cache.pkl")

# 1. Data Science Ingestion & Full Feature Prep
df_raw = load_and_cache_dataset(raw_path=RAW_DATA_PATH, cache_path=CACHE_DATA_PATH)
df_cleaned = clean_and_typecast_data(df_raw)
df_engineered = perform_feature_engineering(df_cleaned)

# 2. AI Preprocessing & Feature Selection
X, y = load_and_preprocess_data(file_path=RAW_DATA_PATH, use_cache=False)

# 3. Data Partitioning: Isolated 80% Training Set & 20% Test Set
X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size=0.20, random_state=42)


# ==============================================================================
# INTEGRATION CHECKPOINTS
# ==============================================================================

def test_load_dataset():
    """
    Checkpoint 1: Validates raw data loading (all 22 features), raw schema contract,
    feature selection filtering (11 of 22 features), schema ordering, feature leak prevention,
    data hygiene (no NaNs in X), and target label domain integrity.
    """
    # 1. Raw Dataset
    assert validate_dataset(df_raw) is True, "Dataset validation contract failed on raw 22-column dataset!"
    assert df_raw.shape[1] == 22, f"Expected exactly 22 raw features in df_raw, but found {df_raw.shape[1]}!"
    assert 'GPA' in df_raw.columns, "Target column 'GPA' missing from raw dataset!"

    # 2. Feature Selection & Preprocessing
    assert len(SELECTED_FEATURES) == 11, f"Expected 11 selected features defined, got {len(SELECTED_FEATURES)}!"
    assert X.shape[1] == 11, f"Expected preprocessed X to have 11 columns, got {X.shape[1]}!"
    
    # Strict Schema Identity & Column Ordering Check
    assert list(X.columns) == list(SELECTED_FEATURES), (
        "Dataset columns do not match expected SELECTED_FEATURES schema or ordering!\n"
        f"Got:      {list(X.columns)}\n"
        f"Expected: {list(SELECTED_FEATURES)}"
    )

    # 3. Data Hygiene Check
    assert not X.isnull().any().any(), "Preprocessed feature matrix X contains missing/NaN values!"

    # 4. Feature Leak Prevention
    raw_feature_cols = [c for c in df_raw.columns if c != 'GPA']
    dropped_features = set(raw_feature_cols) - set(SELECTED_FEATURES)
    
    assert len(dropped_features) == 10, f"Expected exactly 10 features to be dropped, found {len(dropped_features)}!"
    leaked_features = set(X.columns).intersection(dropped_features)
    assert len(leaked_features) == 0, f"Feature selection failed! Leaked raw columns into X: {leaked_features}"

    # 5. Target Label Domain & Cardinality Contract
    assert X.shape[0] == y.shape[0], "Cardinality Mismatch: Row counts between X and y do not match!"
    assert set(np.unique(y)).issubset({1, 2, 3, 4, 5}), "GPA target values must strictly fall within {1, 2, 3, 4, 5}!"


def test_train_test_split():
    """
    Checkpoint 2: Validates isolated 80/20 Stratified Partitioning shapes, ratios,
    sample conservation, column preservation against X, and class coverage.
    """
    total_samples = X.shape[0]
    
    # 1. Sample Conservation & Exact Split Ratios (~80% train, ~20% test)
    assert X_train.shape[0] + X_test.shape[0] == total_samples, "Split sample sum does not equal total dataset rows!"
    assert pytest.approx(X_train.shape[0] / total_samples, abs=0.02) == 0.80, "Train set is not approximately 80% of total data!"
    assert pytest.approx(X_test.shape[0] / total_samples, abs=0.02) == 0.20, "Test set is not approximately 20% of total data!"
    
    # 2. Column Preservation (comparing directly against X)
    assert X_train.shape[1] == X.shape[1], "X_train lost or gained columns during train_test_split!"
    assert X_test.shape[1] == X.shape[1], "X_test lost or gained columns during train_test_split!"
    
    # 3. Label Count & Cardinality Alignment
    assert len(y_train) == X_train.shape[0], "Mismatch between X_train and y_train row counts!"
    assert len(y_test) == X_test.shape[0], "Mismatch between X_test and y_test row counts!"

    # 4. Full Target Class Coverage Check
    assert len(np.unique(y_train)) == 5, "Train split is missing one or more target GPA classes!"
    assert len(np.unique(y_test)) == 5, "Test split is missing one or more target GPA classes!"

    # 5. Stratification Verification
    train_class_props = y_train.value_counts(normalize=True).sort_index()
    full_class_props = y.value_counts(normalize=True).sort_index()
    np.testing.assert_allclose(
        train_class_props.values, 
        full_class_props.values, 
        atol=0.03,
        err_msg="Train split class distribution deviated from target population (stratification failure)!"
    )


def test_cross_validation_setup():
    """
    Checkpoint 3: Validates Stratified 5-Fold Cross-Validation configuration on X_train.
    Guarantees no data leakage into the isolated 20% test set.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(X_train, y_train))
    
    assert len(splits) == 5, "StratifiedKFold failed to generate exactly 5 splits!"
    
    for train_idx, val_idx in splits:
        # Guarantee training + validation samples equal the full training split size
        assert len(train_idx) + len(val_idx) == len(X_train)
        # Guarantee class balance preservation across validation folds
        assert len(np.unique(y_train.iloc[val_idx])) == len(np.unique(y_train))


def test_data_science_and_eda_pipeline():
    """
    Checkpoint 4: End-to-End Data Science & EDA Verification (cbdatsi).
    Validates data quality contracts, label engineering, K-Means student clustering,
    silhouette evaluation, and Chi-Square statistical inference.
    """
    # 1. Structural Validation & Data Hygiene Contract
    assert validate_dataset(df_raw) is True, "Dataset validation contract failed on raw data!"
    
    # 2. Feature Engineering Output Contract
    required_labels = ['GPA_Label', 'Year_Label', 'Gender_Label', 'Poor_Stu_Label', 'Policy_Stu_Label']
    for label in required_labels:
        assert label in df_engineered.columns, f"Engineered column '{label}' missing from df_engineered!"
        assert df_engineered[label].isnull().sum() == 0, f"Feature engineering produced NaN values in '{label}'!"

    # 3. Unsupervised Machine Learning (K-Means Profiling)
    df_clustered, cluster_summary = run_kmeans_clustering(df_engineered, n_clusters=3)
    assert 'Cluster' in df_clustered.columns, "K-Means failed to assign 'Cluster' labels to dataframe!"
    assert set(df_clustered['Cluster'].unique()).issubset({0, 1, 2}), "Unexpected cluster identifiers detected!"
    
    # EDA State & Total Sample Consistency Check
    assert len(df_clustered) == df_raw.shape[0], "Clustered dataframe row count does not match raw dataset shape!"
    
    # Math Bound Check + Quality Gate for Silhouette Score [-1.0, 1.0]
    silhouette = evaluate_clusters(df_clustered)
    assert -1.0 <= silhouette <= 1.0, f"Silhouette score output out of theoretical bounds [-1.0, 1.0]! Got: {silhouette}"
    assert silhouette > 0.10, f"Cluster quality degraded! Silhouette score was {silhouette:.4f}"

    # 4. Statistical Inference & Hypothesis Testing (Chi-Square Test of Independence)
    chi2, p_val, dof, contingency_table = perform_chisquare_independence(
        df_clustered, target_col='GPA', cluster_col='Cluster'
    )
    
    # Contingency Table Sample Conservation Check
    assert contingency_table.shape == (3, 5), "Contingency table shape mismatch (expected 3 Clusters x 5 GPA tiers)!"
    assert contingency_table.values.sum() == df_raw.shape[0], "Contingency table total sample sum mismatch!"

    # Math Bound Checks for Chi-Square outputs
    assert chi2 >= 0.0, f"Chi-Square statistic must be non-negative! Got: {chi2}"
    assert 0.0 <= p_val <= 1.0, f"P-value out of theoretical bounds [0.0, 1.0]! Got: {p_val}"
    
    # Feature Significance Gate
    assert p_val < 0.05, (
        f"Data Science Gate Failure: Student Clusters lost statistical significance with GPA "
        f"(p = {p_val:.4e} >= 0.05)!"
    )


@pytest.mark.parametrize("model_type", ["ordinal_lr", "mlp", "rf"])
def test_full_pipeline_end_to_end(model_type):
    """
    Checkpoints 5 & 6: System Test for Model Training, Inference, and Performance Gates (cbadvai).
    Tests end-to-end execution across Ordinal LR, MLP, and Random Forest architectures
    and validates output against model-specific SLA performance contracts.
    """
    thresholds = MODEL_THRESHOLDS[model_type]
    
    # 1. Build & Train Model Pipeline
    if model_type == "ordinal_lr":
        model_pipeline = build_ordinal_lr_pipeline(random_state=42)
    elif model_type == "mlp":
        model_pipeline = build_mlp_pipeline(random_state=42)
    elif model_type == "rf":
        model_pipeline = build_rf_pipeline(random_state=42)
    else:
        pytest.fail(f"Unrecognized model type: {model_type}")

    model_pipeline.fit(X_train, y_train)
    
    # 2. Execute Inference on Isolated Test Set
    predictions = model_pipeline.predict(X_test)
    
    # 3. Validate Inference Output Contract
    assert len(predictions) == X_test.shape[0], f"[{model_type}] Prediction count mismatch with X_test!"
    assert set(np.unique(predictions)).issubset({1, 2, 3, 4, 5}), f"[{model_type}] Invalid GPA rank predictions detected!"
    
    # 4. Metric Evaluation
    results = evaluate_ordinal_model(y_test, predictions)
    
    # 5. Strict Metric Key Existence Assertions
    assert 'MAE' in results, f"[{model_type}] Evaluation contract broken! Missing 'MAE' key in results."
    assert 'Macro_F1' in results, f"[{model_type}] Evaluation contract broken! Missing 'Macro_F1' key in results."
    assert 'QWK' in results, f"[{model_type}] Evaluation contract broken! Missing 'QWK' key in results."
    
    current_mae = results['MAE']
    current_f1 = results['Macro_F1']
    current_qwk = results['QWK']
    
    # --------------------------------------------------------------------------
    # A. MATHEMATICAL BOUNDS
    # --------------------------------------------------------------------------
    assert current_mae >= 0.0, f"[{model_type}] MAE cannot be negative! Got: {current_mae}"
    assert 0.0 <= current_f1 <= 1.0, f"[{model_type}] Macro F1 out of bounds [0.0, 1.0]! Got: {current_f1}"
    assert -1.0 <= current_qwk <= 1.0, f"[{model_type}] QWK out of bounds [-1.0, 1.0]! Got: {current_qwk}"
    
    # --------------------------------------------------------------------------
    # B. SLA PERFORMANCE GATES
    # --------------------------------------------------------------------------
    assert current_mae <= thresholds['max_mae'], (
        f"[{model_type}] MAE Gate Failed! Got {current_mae:.4f}, required <= {thresholds['max_mae']:.4f}"
    )
    assert current_f1 >= thresholds['min_f1'], (
        f"[{model_type}] Macro F1 Gate Failed! Got {current_f1:.4f}, required >= {thresholds['min_f1']:.4f}"
    )
    assert current_qwk >= thresholds['min_qwk'], (
        f"[{model_type}] QWK Gate Failed! Got {current_qwk:.4f}, required >= {thresholds['min_qwk']:.4f}"
    )
