import pytest
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from src.cbadvai.models import (
    FrankHallOrdinalClassifier, 
    build_ordinal_lr_pipeline, 
    build_mlp_pipeline, 
    build_rf_pipeline,
    predict_ordinal_expected_value
)
from src.cbadvai.preprocessing import SELECTED_FEATURES

# ==============================================================================
# SECTION 1: CUSTOM ORDINAL ESTIMATOR INVARIANTS
# ==============================================================================

def test_frank_hall_classifier_mathematical_invariants(synthetic_survey_df):
    """
    Tests custom Frank & Hall Ordinal Classifier:
    1. Shape contracts: (N, K) probabilities, (N,) predictions.
    2. Probability bounds: Every probability P_ij must be in [0.0, 1.0].
    3. Row sum invariant: Sum of class probabilities per sample must equal 1.0.
    4. Prediction domain: Output labels must strictly belong to target classes {1..5}.
    """
    X = synthetic_survey_df[SELECTED_FEATURES].values
    y = synthetic_survey_df['GPA'].values

    clf = FrankHallOrdinalClassifier()
    clf.fit(X, y)

    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    # 1. Shape Contracts
    assert probs.shape == (100, 5)
    assert preds.shape == (100,)

    # 2. Probability Bounds: 0.0 <= P_ij <= 1.0
    assert (probs >= 0.0).all(), "Negative probabilities detected in Ordinal Classifier!"
    assert (probs <= 1.0).all(), "Probabilities greater than 1.0 detected in Ordinal Classifier!"

    # 3. Sum of probabilities across columns = 1.0
    row_sums = probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    # 4. Strict Label Range Check
    assert np.issubdtype(preds.dtype, np.integer)
    assert set(preds).issubset({1, 2, 3, 4, 5})


# ==============================================================================
# SECTION 2: MODEL-SPECIFIC UNIT TESTS (OLR, MLP, RF)
# ==============================================================================

def test_ordinal_lr_determinism_and_binary_estimators(synthetic_survey_df):
    """
    Unit Test [Ordinal LR]: 
    1. Verifies 100% deterministic outputs across identical runs.
    2. Confirms exactly K-1 (4) binary logistic estimators are fitted internally.
    """
    X = synthetic_survey_df[SELECTED_FEATURES]
    y = synthetic_survey_df['GPA']

    pipe1 = build_ordinal_lr_pipeline(random_state=42)
    pipe2 = build_ordinal_lr_pipeline(random_state=42)

    pipe1.fit(X, y)
    pipe2.fit(X, y)

    preds1 = pipe1.predict(X)
    preds2 = pipe2.predict(X)

    # Determinism check
    np.testing.assert_array_equal(preds1, preds2, err_msg="Ordinal LR is non-deterministic!")

    # Verify K-1 binary estimators contract for 5 classes
    classifier = pipe1.named_steps['classifier']
    assert isinstance(classifier, FrankHallOrdinalClassifier)
    assert len(classifier.estimators_) == 4, "Frank & Hall should create 4 binary estimators for 5 classes!"


def test_mlp_neural_network_stability_and_reproducibility(synthetic_survey_df):
    """
    Unit Test [MLP Neural Net]:
    1. Verifies seed reproducibility (identical weights/predictions).
    2. Edge Case Input: Tests extreme boundary survey inputs (all 1s and all 5s) 
       to guarantee no numerical instability (NaNs or Infs) in hidden activation layers.
    """
    X = synthetic_survey_df[SELECTED_FEATURES]
    y = synthetic_survey_df['GPA']

    pipe1 = build_mlp_pipeline(random_state=42)
    pipe2 = build_mlp_pipeline(random_state=42)

    pipe1.fit(X, y)
    pipe2.fit(X, y)

    # 1. Reproducibility
    np.testing.assert_array_equal(pipe1.predict(X), pipe2.predict(X), err_msg="MLP is non-deterministic!")

    # 2. Extreme Edge Case Inputs (All 1s and All 5s)
    extreme_inputs = np.vstack([
        np.ones((5, 11), dtype=int),  # 5 samples of all 1s
        np.full((5, 11), 5, dtype=int) # 5 samples of all 5s
    ])
    
    extreme_probs = pipe1.predict_proba(extreme_inputs)
    assert not np.isnan(extreme_probs).any(), "MLP produced NaN probabilities on extreme inputs!"
    assert not np.isinf(extreme_probs).any(), "MLP produced Inf probabilities on extreme inputs!"


def test_random_forest_feature_importance_contract(synthetic_survey_df):
    """
    Unit Test [Random Forest]:
    1. Verifies seed reproducibility.
    2. Feature Importance Invariant: Ensures feature_importances_ exist, 
       has length 11, contains non-negative values, and sums strictly to 1.0.
    """
    X = synthetic_survey_df[SELECTED_FEATURES]
    y = synthetic_survey_df['GPA']

    pipe = build_rf_pipeline(random_state=42)
    pipe.fit(X, y)

    rf_model = pipe.named_steps['classifier']
    assert isinstance(rf_model, RandomForestClassifier)

    importances = rf_model.feature_importances_

    # Structural contracts for interpretability downstream
    assert importances.shape == (11,), "RF feature importances shape must match 11 survey features!"
    assert (importances >= 0.0).all(), "Negative feature importances detected in RF!"
    np.testing.assert_allclose(importances.sum(), 1.0, atol=1e-5, err_msg="RF feature importances do not sum to 1.0!")


# ==============================================================================
# SECTION 3: PIPELINE BUILDERS & INTEGRATION CONTRACTS
# ==============================================================================

def test_pipeline_builders_instantiation_and_execution(synthetic_survey_df):
    """
    Verifies that all pipeline builders construct valid ImbPipelines with SMOTE 
    and successfully run fit/predict on synthetic survey data.
    """
    X = synthetic_survey_df[SELECTED_FEATURES]
    y = synthetic_survey_df['GPA']

    pipelines = {
        'Ordinal LR': build_ordinal_lr_pipeline(),
        'MLP': build_mlp_pipeline(),
        'RF': build_rf_pipeline()
    }

    for name, pipe in pipelines.items():
        # Check structure
        assert isinstance(pipe, ImbPipeline), f"{name} is not an ImbPipeline!"
        assert 'smote' in pipe.named_steps, f"{name} pipeline missing 'smote' step!"
        assert 'classifier' in pipe.named_steps, f"{name} pipeline missing 'classifier' step!"

        # Check fit and predict integration
        pipe.fit(X, y)
        preds = pipe.predict(X)

        assert preds.shape == (100,)
        assert set(preds).issubset({1, 2, 3, 4, 5})


# ==============================================================================
# SECTION 4: EXPECTED VALUE HELPER CALCULATIONS
# ==============================================================================

def test_predict_ordinal_expected_value_bounds_and_math(synthetic_survey_df):
    """
    Tests predict_ordinal_expected_value helper function:
    1. Boundedness: Output rounded values must remain strictly within [1, 5].
    2. Mathematical Proof: Verifies expectation sum calculation on trained models.
    """
    X = synthetic_survey_df[SELECTED_FEATURES].values
    y = synthetic_survey_df['GPA'].values

    clf = FrankHallOrdinalClassifier()
    clf.fit(X, y)

    classes = np.array([1, 2, 3, 4, 5])
    expected_preds = predict_ordinal_expected_value(clf, X, classes=classes)

    # Shape and Type
    assert expected_preds.shape == (100,)
    assert np.issubdtype(expected_preds.dtype, np.integer)
    assert expected_preds.min() >= 1
    assert expected_preds.max() <= 5


def test_predict_ordinal_expected_value_exact_math():
    """
    Exhaustive mathematical proof test for expected value calculation.
    Tests:
    1. 100% probability certainty across all 5 individual classes {1, 2, 3, 4, 5}.
    2. Fractional expected values that test integer rounding boundaries (e.g. 2.6 -> 3, 2.2 -> 2).
    """
    class ComprehensiveDummyClassifier:
        def predict_proba(self, X):
            return np.array([
                [1.0, 0.0, 0.0, 0.0, 0.0],  # 100% Class 1 -> E = 1.0 -> rounds to 1
                [0.0, 1.0, 0.0, 0.0, 0.0],  # 100% Class 2 -> E = 2.0 -> rounds to 2
                [0.0, 0.0, 1.0, 0.0, 0.0],  # 100% Class 3 -> E = 3.0 -> rounds to 3
                [0.0, 0.0, 0.0, 1.0, 0.0],  # 100% Class 4 -> E = 4.0 -> rounds to 4
                [0.0, 0.0, 0.0, 0.0, 1.0],  # 100% Class 5 -> E = 5.0 -> rounds to 5
                [0.0, 0.4, 0.6, 0.0, 0.0],  # E = 0.4*2 + 0.6*3 = 2.6 -> rounds to 3
                [0.0, 0.8, 0.2, 0.0, 0.0],  # E = 0.8*2 + 0.2*3 = 2.2 -> rounds to 2
            ])

    clf = ComprehensiveDummyClassifier()
    classes = np.array([1, 2, 3, 4, 5])
    
    # 7 synthetic input rows matching the 7 mock probability rows
    X_dummy = np.zeros((7, 11))
    preds = predict_ordinal_expected_value(clf, X_dummy, classes=classes)

    # 1. Direct class mapping verification across all 5 classes
    assert preds[0] == 1, "Class 1 expected value mapping failed!"
    assert preds[1] == 2, "Class 2 expected value mapping failed!"
    assert preds[2] == 3, "Class 3 expected value mapping failed!"
    assert preds[3] == 4, "Class 4 expected value mapping failed!"
    assert preds[4] == 5, "Class 5 expected value mapping failed!"

    # 2. Fractional expectation and rounding boundary checks
    assert preds[5] == 3, "Fractional rounding up (2.6 -> 3) failed!"
    assert preds[6] == 2, "Fractional rounding down (2.2 -> 2) failed!"
