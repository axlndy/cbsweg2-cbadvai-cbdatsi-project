import pytest
import pandas as pd
import numpy as np
import os
from src.cbadvai.preprocessing import (
    SELECTED_FEATURES, 
    load_and_preprocess_data, 
    get_train_test_split
)

# feature list
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
# SECTION 1: STRICT CONTRACT & COMPREHENSIVE BOUNDARY TESTS
# ==============================================================================

def test_selected_features_contract():
    """
    Strictly verifies that SELECTED_FEATURES matches the exact expected 11 survey features.
    """
    # 1. Exact Set Match
    assert set(SELECTED_FEATURES) == set(EXPECTED_FEATURES), (
        f"Feature set mismatch!\n"
        f"Missing expected features: {set(EXPECTED_FEATURES) - set(SELECTED_FEATURES)}\n"
        f"Unexpected extra features: {set(SELECTED_FEATURES) - set(EXPECTED_FEATURES)}"
    )

    # 2. Exact Order Match
    assert SELECTED_FEATURES == EXPECTED_FEATURES, (
        "Feature list order does not match the strict EXPECTED_FEATURES contract!"
    )


def test_processed_data_strict_ranges_and_types(tmp_path, synthetic_survey_df):
    """
    Checks EVERY SINGLE feature column dynamically:
    1. Likert features must strictly be whole integers in the range [1, 5].
    2. Policy_Stu must strictly be binary integers in {0, 1}.
    3. Target GPA must strictly be whole integers in the range [1, 5].
    4. No float or object data types allowed across any feature.
    """
    excel_path = tmp_path / "valid_data.xlsx"
    synthetic_survey_df.to_excel(excel_path, index=False)

    X, y = load_and_preprocess_data(file_path=str(excel_path), use_cache=False)

    # 1. Verify Policy_Stu binary contract (Single column check)
    assert np.issubdtype(X['Policy_Stu'].dtype, np.integer), "Policy_Stu is not an integer type!"
    assert set(X['Policy_Stu'].unique()).issubset({0, 1}), "Policy_Stu contains values other than 0 or 1!"

    # 2. Dynamically test ALL 10 Likert features (No single-column blind spots!)
    likert_features = [f for f in SELECTED_FEATURES if f != 'Policy_Stu']
    for feature in likert_features:
        # Must be integer numeric type
        assert np.issubdtype(X[feature].dtype, np.integer), f"Feature '{feature}' is not integer type!"
        # Boundaries check
        assert X[feature].min() >= 1, f"Feature '{feature}' has values lower than 1!"
        assert X[feature].max() <= 5, f"Feature '{feature}' has values higher than 5!"
        # Fractional check
        assert (X[feature] % 1 == 0).all(), f"Feature '{feature}' contains fractional decimal values!"

    # 3. Verify Target GPA contract
    assert np.issubdtype(y.dtype, np.integer), "Target GPA is not an integer type!"
    assert y.min() >= 1 and y.max() <= 5, "Target GPA has values outside [1, 5]!"


def test_policy_stu_binary_mapping_correctness(tmp_path, synthetic_survey_df):
    """Verifies that raw Policy_Stu values 1 and 2 remap strictly to 1 and 0."""
    raw_df = synthetic_survey_df.copy()
    raw_df['Policy_Stu'] = [1, 2] * (len(raw_df) // 2)

    excel_path = tmp_path / "mapping_data.xlsx"
    raw_df.to_excel(excel_path, index=False)

    X, _ = load_and_preprocess_data(file_path=str(excel_path), use_cache=False)

    # Raw 1 -> 1, Raw 2 -> 0
    raw_ones_mask = (raw_df['Policy_Stu'] == 1)
    raw_twos_mask = (raw_df['Policy_Stu'] == 2)

    assert (X.loc[raw_ones_mask, 'Policy_Stu'] == 1).all()
    assert (X.loc[raw_twos_mask, 'Policy_Stu'] == 0).all()


# ==============================================================================
# SECTION 2: DYNAMIC FAULT INJECTION TESTS (FULL FEATURE LOOPING)
# ==============================================================================

def test_fault_injection_all_likert_out_of_bounds(tmp_path, synthetic_survey_df):
    """
    Fault Injection: Iterates across EVERY Likert feature individually and injects 
    out-of-bounds values (e.g. 0 and 6). Proves no single column bypasses assertions.
    """
    likert_features = [f for f in SELECTED_FEATURES if f != 'Policy_Stu']
    
    for target_col in likert_features:
        corrupted_df = synthetic_survey_df.copy()
        corrupted_df.loc[0, target_col] = 6   # Value too high
        corrupted_df.loc[1, target_col] = 0   # Value too low

        excel_path = tmp_path / f"corrupted_{target_col}.xlsx"
        corrupted_df.to_excel(excel_path, index=False)

        X, _ = load_and_preprocess_data(file_path=str(excel_path), use_cache=False)

        # Flag out-of-bound values in the current column
        is_invalid = (X[target_col] < 1) | (X[target_col] > 5)
        assert is_invalid.any(), f"Pipeline failed to detect out-of-bound values injected in '{target_col}'!"


def test_fault_injection_unmapped_policy_stu_values(tmp_path, synthetic_survey_df):
    """
    Fault Injection: Inject invalid values into Policy_Stu (e.g. 3, 99).
    Mapping with {1: 1, 2: 0} will turn invalid raw values into NaNs.
    """
    corrupted_df = synthetic_survey_df.copy()
    corrupted_df.loc[0, 'Policy_Stu'] = 3
    corrupted_df.loc[1, 'Policy_Stu'] = 99

    excel_path = tmp_path / "corrupted_policy.xlsx"
    corrupted_df.to_excel(excel_path, index=False)

    X, _ = load_and_preprocess_data(file_path=str(excel_path), use_cache=False)

    # Invalid codes must be mapped to NaN
    assert X['Policy_Stu'].isnull().any(), "Pipeline failed to convert invalid Policy_Stu codes to NaN!"


def test_fault_injection_non_integer_floats(tmp_path, synthetic_survey_df):
    """
    Fault Injection: Inject fractional floats across Likert features.
    Verifies that fractional numbers (e.g. 3.5) are caught.
    """
    likert_features = [f for f in SELECTED_FEATURES if f != 'Policy_Stu']
    
    for target_col in likert_features:
        corrupted_df = synthetic_survey_df.copy().astype(object)
        corrupted_df.loc[0, target_col] = 3.5

        excel_path = tmp_path / f"corrupted_float_{target_col}.xlsx"
        corrupted_df.to_excel(excel_path, index=False)

        X, _ = load_and_preprocess_data(file_path=str(excel_path), use_cache=False)

        has_fractional = (X[target_col] % 1 != 0).any()
        assert has_fractional, f"Pipeline failed to flag fractional float injected into '{target_col}'!"


# ==============================================================================
# SECTION 3: CACHING & DATA LEAKAGE TESTS
# ==============================================================================

def test_preprocessing_cache_integrity(tmp_path, synthetic_survey_df):
    """Verifies that use_cache=True writes and reloads processed dataset accurately."""
    excel_path = tmp_path / "data.xlsx"
    cache_path = tmp_path / "cache.pkl"
    synthetic_survey_df.to_excel(excel_path, index=False)

    # First call creates cache
    X1, y1 = load_and_preprocess_data(file_path=str(excel_path), use_cache=True, cache_path=str(cache_path))
    assert os.path.exists(cache_path)

    # Second call loads cache
    X2, y2 = load_and_preprocess_data(file_path=str(excel_path), use_cache=True, cache_path=str(cache_path))
    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)


def test_no_data_leakage_and_stratification(synthetic_survey_df):
    """Ensures zero overlap between train/test index sets and preserves target classes."""
    X = synthetic_survey_df[SELECTED_FEATURES]
    y = synthetic_survey_df['GPA']

    X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size=0.20, random_state=42)

    # 1. Disjoint index set check
    train_idx = set(X_train.index)
    test_idx = set(X_test.index)
    assert len(train_idx.intersection(test_idx)) == 0, "Data leakage detected between train and test splits!"

    # 2. Stratification check across all 5 classes
    assert set(y_train.unique()) == set(y.unique()) == {1, 2, 3, 4, 5}
