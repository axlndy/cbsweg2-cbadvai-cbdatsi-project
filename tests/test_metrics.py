import pytest
import numpy as np
import pandas as pd
from src.cbadvai.metrics import evaluate_ordinal_model, compute_summary_feature_weights
from src.cbadvai.preprocessing import SELECTED_FEATURES
from src.cbadvai.models import build_ordinal_lr_pipeline, build_rf_pipeline

# Canonical 11 feature contract
EXPECTED_FEATURES = [
    'Study_Methods', 
    'Time_Studying', 
    'Time_Friends', 
    'Time_SocicalMedia', 
    'Adapt_Learning_Uni', 
    'Policy_Stu', 
    'SupportOf_Uni', 
    'SupportOf_Lec', 
    'Facilitie_Uni', 
    'Quality_Lecturer', 
    'TrainingCurriculum'
]

# ==============================================================================
# SECTION 1: EVALUATION METRICS CONTRACTS & BOUNDS
# ==============================================================================

def test_evaluate_ordinal_model_dictionary_and_bounds():
    """
    Tests evaluation function dictionary structure, metrics math, and strict domain bounds:
    1. Macro_F1 must be strictly within [0.0, 1.0].
    2. MAE must be strictly within [0.0, 4.0] for a 1-to-5 GPA scale (|5 - 1| = 4.0).
    3. QWK must be strictly within [-1.0, 1.0].
    4. Confusion matrix must conserve sample counts (cm.sum() == N).
    """
    y_true = np.array([1, 2, 3, 4, 5, 3, 2, 4, 5, 1])
    y_pred = np.array([1, 2, 3, 4, 4, 3, 2, 4, 5, 2])

    metrics = evaluate_ordinal_model(y_true, y_pred, model_name="Test Model")

    # Dict keys contract
    for key in ['Macro_F1', 'MAE', 'QWK', 'Confusion_Matrix']:
        assert key in metrics, f"Missing key '{key}' in evaluate_ordinal_model dictionary!"

    # Numerical range checks (Strict Domain Bounds)
    assert 0.0 <= metrics['Macro_F1'] <= 1.0, f"Macro_F1 out of bounds: {metrics['Macro_F1']}"
    assert 0.0 <= metrics['MAE'] <= 4.0, f"MAE out of domain bounds (> 4.0): {metrics['MAE']}"
    assert -1.0 <= metrics['QWK'] <= 1.0, f"QWK out of bounds: {metrics['QWK']}"

    # Confusion matrix contract
    cm = metrics['Confusion_Matrix']
    assert isinstance(cm, np.ndarray)
    assert cm.shape == (5, 5)
    assert (cm >= 0).all()
    
    # Sample Conservation Check
    assert cm.sum() == len(y_true), f"Confusion matrix total {cm.sum()} != sample count {len(y_true)}"


def test_evaluate_ordinal_model_perfect_and_worst_case_edge_cases():
    """
    Edge Cases:
    1. Perfect predictions: QWK = 1.0, MAE = 0.0, Macro_F1 = 1.0.
    2. True Worst-case predictions (1 -> 5, 5 -> 1): MAE = 4.0, Macro_F1 = 0.0, QWK = -1.0.
    """
    # 1. Perfect Match Test (Safe Float Checks)
    y_true_perf = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    perf_metrics = evaluate_ordinal_model(y_true_perf, y_true_perf.copy(), model_name="Perfect Model")
    
    assert perf_metrics['Macro_F1'] == pytest.approx(1.0)
    assert perf_metrics['MAE'] == pytest.approx(0.0)
    assert perf_metrics['QWK'] == pytest.approx(1.0)

    # 2. Maximum Worst-Case Test (All 1s predicted as 5s, all 5s predicted as 1s)
    y_true_worst  = np.array([1, 1, 1, 1, 5, 5, 5, 5])
    y_pred_worst  = np.array([5, 5, 5, 5, 1, 1, 1, 1])
    
    worst_metrics = evaluate_ordinal_model(y_true_worst, y_pred_worst, model_name="Worst Model")
    
    assert worst_metrics['MAE'] == pytest.approx(4.0)       # Max possible absolute error on 1-5 scale
    assert worst_metrics['Macro_F1'] == pytest.approx(0.0)  # Zero precision/recall
    assert worst_metrics['QWK'] == pytest.approx(-1.0)     # Maximum possible disagreement


def test_evaluate_ordinal_model_unseen_classes_and_mismatched_inputs():
    """
    Edge Cases:
    1. Full Class Coverage: Verifies a 5x5 matrix is returned when all GPA classes (1-5) are present.
    2. Dimension mismatch: Raises ValueError when y_true and y_pred lengths differ.
    """
    # 1. Inputs containing all 5 GPA categories (reflecting real notebook evaluation)
    y_true_full = np.array([1, 2, 3, 4, 5])
    y_pred_full = np.array([1, 2, 3, 4, 4])
    
    full_metrics = evaluate_ordinal_model(y_true_full, y_pred_full, model_name="Full Model")
    
    assert full_metrics['Confusion_Matrix'].shape == (5, 5)

    with pytest.raises((ValueError, AssertionError)):
        evaluate_ordinal_model(np.array([1, 2, 3]), np.array([1, 2]), model_name="Broken Input")


# ==============================================================================
# SECTION 2: FEATURE WEIGHT SUMMARY & NORMALIZATION INVARIANTS
# ==============================================================================

def test_compute_summary_feature_weights_normalization(synthetic_survey_df):
    """
    Verifies feature weight extraction and normalization:
    1. Output DataFrame must strictly have 11 rows matching EXPECTED_FEATURES in order.
    2. Column structure must match expected model labels.
    3. Individual weight entries must strictly be within [0.0, 1.0] without NaNs.
    4. Sum of normalized weights per model column must strictly equal 1.0.
    """
    X = synthetic_survey_df[SELECTED_FEATURES]
    y = synthetic_survey_df['GPA']

    pipe_lr = build_ordinal_lr_pipeline()
    pipe_lr.fit(X, y)

    pipe_rf1 = build_rf_pipeline()
    pipe_rf1.fit(X, y)

    pipe_rf2 = build_rf_pipeline()
    pipe_rf2.fit(X, y)

    weights_df = compute_summary_feature_weights(
        pipe_lr, pipe_lr, pipe_rf1, pipe_rf2, 
        X, y, feature_names=SELECTED_FEATURES
    )

    assert isinstance(weights_df, pd.DataFrame)
    
    # 1. Exact Row Count and Index Alignment Check
    assert len(weights_df) == 11
    assert list(weights_df.index) == EXPECTED_FEATURES

    # 2. Column Count Invariant (4 models passed -> 4 weight columns)
    assert len(weights_df.columns) == 4, f"Expected 4 weight columns, got {len(weights_df.columns)}"

    # 3. Comprehensive Value Range & Normalization Checks
    for col in weights_df.columns:
        # A. Non-null check
        assert not weights_df[col].isnull().any(), f"Column '{col}' contains NaN values!"
        
        # B. Individual Cell Bounds Check: 0.0 <= w_i <= 1.0
        assert (weights_df[col] >= 0.0).all(), f"Column '{col}' contains negative feature weights!"
        assert (weights_df[col] <= 1.0).all(), f"Column '{col}' contains weights greater than 1.0!"
        
        # C. Column Sum Check: sum(w_i) == 1.0
        col_sum = weights_df[col].sum()
        np.testing.assert_allclose(col_sum, 1.0, atol=1e-4, err_msg=f"Column '{col}' weights do not sum to 1.0!")
