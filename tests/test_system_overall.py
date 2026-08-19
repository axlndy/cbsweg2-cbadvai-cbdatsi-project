# tests/test_system_overall.py
"""
End-to-End System Tests for EduPredict.

Verifies integration across:
    CBDATSI:
        Load -> Validate -> Clean -> Feature Engineering
        -> EDA -> K-Means -> Chi-Square Inference

    CBADVAI:
        Load -> Feature Selection -> 80/20 Stratified Split
        -> Actual 5-Fold Model Selection
        -> Initial Test-Set Evaluation
        -> Improved RF Evaluation
        -> Feature Weight Analysis

The system tests intentionally use performance ranges rather than
requiring exact reproduction of floating-point notebook values.
"""

import os

import numpy as np
import pandas as pd
import pytest

from sklearn.model_selection import StratifiedKFold

# ==============================================================================
# CBDATSI IMPORTS
# ==============================================================================

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

from src.cbdatsi.inference import (
    perform_chisquare_independence,
)

from src.cbdatsi.eda_plots import (
    plot_demographics,
    plot_behavioral_boxplots,
    plot_socioeconomic_conditional,
    plot_institutional_heatmap,
    plot_mindset_heatmaps,
)


# ==============================================================================
# CBADVAI IMPORTS
# ==============================================================================

from src.cbadvai.preprocessing import (
    load_and_preprocess_data,
    get_train_test_split,
    SELECTED_FEATURES,
)

from src.cbadvai.models import (
    tune_ordinal_lr,
    tune_mlp,
    tune_initial_rf,
    tune_improved_rf,
    predict_ordinal_expected_value,
)

from src.cbadvai.metrics import (
    evaluate_ordinal_model,
    compute_summary_feature_weights,
)


# ==============================================================================
# PROJECT PATHS
# ==============================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RAW_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "Database paper.xlsx",
)

CACHE_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "dataset_cache.pkl",
)


# ==============================================================================
# EXPECTED FEATURE CONTRACTS
# ==============================================================================

EXPECTED_CBADVAI_FEATURES = [
    "Study_Methods",
    "Time_Studying",
    "Time_Friends",
    "Time_SocicalMedia",
    "Adapt_Learning_Uni",
    "Policy_Stu",
    "SupportOf_Uni",
    "SupportOf_Lec",
    "Facilitie_Uni",
    "Quality_Lecturer",
    "TrainingCurriculum",
]


EXPECTED_CBDATSI_LABELS = [
    "GPA_Label",
    "Year_Label",
    "Gender_Label",
    "Poor_Stu_Label",
    "Policy_Stu_Label",
]


# ==============================================================================
# CBADVAI PERFORMANCE GATES
# ==============================================================================

# These are intentionally less strict than exact notebook values.
#
# Notebook baseline:
#   Initial OLR:  F1 ~0.2216, MAE ~1.1475
#   Initial MLP:  F1 ~0.2545, MAE ~0.9447
#   Initial RF:   F1 ~0.3084, MAE ~0.6382
#
# Improved RF:
#   F1 ~0.3130, MAE ~0.6221, QWK ~0.2257
#
# We test that performance remains reasonably close to the established
# analysis rather than requiring exact floating-point reproduction.

INITIAL_MODEL_GATES = {
    "Ordinal LR": {
        "min_f1": 0.18,
        "max_mae": 1.38,
    },
    "Optimized MLP": {
        "min_f1": 0.21,
        "max_mae": 1.14,
    },
    "Initial RF": {
        "min_f1": 0.26,
        "max_mae": 0.77,
    },
}

IMPROVED_RF_GATE = {
    "min_f1": 0.27,
    "max_mae": 0.75,
    "min_qwk": 0.18,
}


# ==============================================================================
# GLOBAL DATA PREPARATION
# ==============================================================================

df_raw = load_and_cache_dataset(
    raw_path=RAW_DATA_PATH,
    cache_path=CACHE_DATA_PATH,
)

df_cleaned = clean_and_typecast_data(df_raw)

df_engineered = perform_feature_engineering(
    df_cleaned
)

X, y = load_and_preprocess_data(
    file_path=RAW_DATA_PATH,
    use_cache=False,
)

X_train, X_test, y_train, y_test = get_train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)


# ==============================================================================
# SHARED CBADVAI MODEL-SELECTION FIXTURE
# ==============================================================================

@pytest.fixture(scope="session")
def cbadvai_model_selection():
    """
    Runs the ACTUAL CBADVAI model-selection pipeline using
    Stratified 5-Fold Cross-Validation.

    This fixture is session-scoped so that the expensive tuning
    process is executed once and reused by downstream system tests.
    """

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # --------------------------------------------------------------
    # 1. Ordinal Logistic Regression
    # --------------------------------------------------------------

    grid_lr = tune_ordinal_lr(
        X_train,
        y_train,
        cv=cv,
    )

    # --------------------------------------------------------------
    # 2. MLP architecture/activation selection
    # --------------------------------------------------------------

    grid_mlp, mlp_results, architecture_keys, best_mlp = tune_mlp(
        X_train,
        y_train,
        cv=cv,
    )

    # --------------------------------------------------------------
    # 3. Initial Random Forest
    # --------------------------------------------------------------

    grid_rf_initial = tune_initial_rf(
        X_train,
        y_train,
        cv=cv,
    )

    # --------------------------------------------------------------
    # 4. Improved Random Forest
    # --------------------------------------------------------------

    grid_rf_improved = tune_improved_rf(
        X_train,
        y_train,
        cv=cv,
    )

    return {
        "cv": cv,
        "grid_lr": grid_lr,
        "grid_mlp": grid_mlp,
        "mlp_results": mlp_results,
        "architecture_keys": architecture_keys,
        "best_mlp": best_mlp,
        "grid_rf_initial": grid_rf_initial,
        "grid_rf_improved": grid_rf_improved,
    }


# ==============================================================================
# CBDATSI SYSTEM TESTS
# ==============================================================================

def test_cbdatsi_load_and_preprocessing():
    """
    Checkpoint 1:
    Raw data successfully passes loading, validation, cleaning,
    and feature engineering.
    """

    # Raw dataset exists and is structurally valid.
    assert os.path.exists(RAW_DATA_PATH)

    assert validate_dataset(df_raw) is True

    # Established dataset size.
    assert len(df_raw) == 2170

    # Cleaning must preserve the complete observation count.
    assert len(df_cleaned) == len(df_raw)

    # Feature engineering must preserve observation count.
    assert len(df_engineered) == len(df_raw)

    # Required engineered labels must exist.
    for label in EXPECTED_CBDATSI_LABELS:
        assert label in df_engineered.columns

        assert not df_engineered[label].isnull().any(), (
            f"Feature engineering produced missing values in {label}."
        )


def test_cbdatsi_eda_integration():
    """
    Checkpoint 2:
    All CBDATSI EDA plotting functions execute successfully
    using the engineered dataset.

    This checks actual integration with the plotting functions,
    rather than only testing them independently with dummy data.
    """

    import matplotlib.pyplot as plt

    plot_demographics(df_engineered)
    plot_behavioral_boxplots(df_engineered)
    plot_socioeconomic_conditional(df_engineered)
    plot_institutional_heatmap(df_engineered)
    plot_mindset_heatmaps(df_engineered)

    # At least one figure must have been created.
    assert len(plt.get_fignums()) > 0

    plt.close("all")


def test_cbdatsi_clustering_integration():
    """
    Checkpoint 3:
    Feature-engineered CBDATSI data flows into K-Means and
    produces valid three-cluster output.
    """

    clustered_df, cluster_summary = run_kmeans_clustering(
        df_engineered,
        n_clusters=3,
    )

    # Row conservation.
    assert len(clustered_df) == len(df_engineered)

    # Cluster assignment exists.
    assert "Cluster" in clustered_df.columns

    # Exactly three clusters must be produced.
    assert clustered_df["Cluster"].nunique() == 3

    # Cluster IDs must be valid.
    assert set(
        clustered_df["Cluster"].unique()
    ).issubset({0, 1, 2})

    # Summary must contain three cluster rows.
    assert len(cluster_summary) == 3

    # Silhouette score must remain theoretically valid.
    silhouette = evaluate_clusters(
        clustered_df
    )

    assert -1.0 <= silhouette <= 1.0

    # Established CBDATSI analysis showed positive cluster quality.
    assert silhouette > 0.10


def test_cbdatsi_inference_and_output():
    """
    Checkpoint 4:
    K-Means output flows into Chi-Square inference and produces
    a valid contingency table and statistically meaningful result.
    """

    clustered_df, _ = run_kmeans_clustering(
        df_engineered,
        n_clusters=3,
    )

    chi2, p_value, dof, contingency_table = (
        perform_chisquare_independence(
            clustered_df,
            target_col="GPA",
            cluster_col="Cluster",
        )
    )

    # Mathematical bounds.
    assert chi2 >= 0.0
    assert 0.0 <= p_value <= 1.0
    assert dof >= 0

    # 3 clusters × 5 GPA levels.
    assert contingency_table.shape == (3, 5)

    # Sample conservation.
    assert contingency_table.values.sum() == len(
        clustered_df
    )

    # Established analysis found significant association.
    assert p_value < 0.05


# ==============================================================================
# CBADVAI PREPROCESSING SYSTEM TEST
# ==============================================================================

def test_cbadvai_preprocessing_and_split():
    """
    Checkpoint 5:
    Verifies feature selection, leakage prevention, target integrity,
    and isolated 80/20 stratified holdout testing.
    """

    # Exactly 11 modeling features.
    assert len(SELECTED_FEATURES) == 11

    assert list(SELECTED_FEATURES) == (
        EXPECTED_CBADVAI_FEATURES
    )

    assert X.shape[1] == 11
    assert list(X.columns) == EXPECTED_CBADVAI_FEATURES

    # No feature missing values.
    assert not X.isnull().any().any()

    # Target cardinality.
    assert len(X) == len(y)

    # GPA domain.
    assert set(
        np.unique(y)
    ).issubset({1, 2, 3, 4, 5})

    # No target leakage.
    assert "GPA" not in X.columns

    # Split conservation.
    assert (
        len(X_train) + len(X_test)
        == len(X)
    )

    assert (
        len(y_train) + len(y_test)
        == len(y)
    )

    # Approximately 80/20.
    assert (
        X_train.shape[0] / X.shape[0]
        == pytest.approx(0.80, abs=0.02)
    )

    assert (
        X_test.shape[0] / X.shape[0]
        == pytest.approx(0.20, abs=0.02)
    )

    # X/y alignment.
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    # Every GPA class must remain represented.
    assert set(np.unique(y_train)) == {
        1, 2, 3, 4, 5
    }

    assert set(np.unique(y_test)) == {
        1, 2, 3, 4, 5
    }

    # Explicit train/test overlap checks.
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)

    assert train_indices.isdisjoint(
        test_indices
    ), "Training and test observations overlap."


# ==============================================================================
# CBADVAI ACTUAL 5-FOLD MODEL SELECTION
# ==============================================================================

def test_cbadvai_actual_five_fold_model_selection(
    cbadvai_model_selection,
):
    """
    Checkpoint 6:
    Verifies that the ACTUAL model-selection pipeline successfully
    performs Stratified 5-Fold Cross-Validation.

    This is deliberately not a fake/manual split test.
    The real tuning functions are executed with a 5-fold CV object.
    """

    results = cbadvai_model_selection

    cv = results["cv"]

    assert cv.n_splits == 5
    assert cv.shuffle is True
    assert cv.random_state == 42

    # Verify the actual folds.
    splits = list(
        cv.split(X_train, y_train)
    )

    assert len(splits) == 5

    # Every observation appears in validation exactly once.
    validation_indices = np.concatenate(
        [val_idx for _, val_idx in splits]
    )

    assert len(validation_indices) == len(X_train)

    assert len(
        np.unique(validation_indices)
    ) == len(X_train)

    # ------------------------------------------------------------------
    # OLR
    # ------------------------------------------------------------------

    grid_lr = results["grid_lr"]

    assert hasattr(grid_lr, "cv_results_")
    assert hasattr(grid_lr, "best_estimator_")

    assert grid_lr.n_splits_ == 5
    assert len(grid_lr.cv_results_["mean_test_score"]) > 0

    # 4 C values × 2 penalties = 8 candidates.
    assert len(
        grid_lr.cv_results_["params"]
    ) == 8

    # ------------------------------------------------------------------
    # MLP
    # ------------------------------------------------------------------

    mlp_results = results["mlp_results"]

    assert isinstance(mlp_results, pd.DataFrame)

    # 7 architectures × 3 activations.
    assert len(mlp_results) == 21

    assert set(
        mlp_results["Activation"]
    ) == {
        "relu",
        "tanh",
        "logistic",
    }

    assert mlp_results["Mean F1"].notna().all()

    assert np.isfinite(
        mlp_results["Mean F1"]
    ).all()

    assert (
        mlp_results["Mean F1"]
        .between(0.0, 1.0)
        .all()
    )

    # Best MLP must actually correspond to the maximum CV score.
    best_mlp = results["best_mlp"]

    assert (
        best_mlp["Mean F1"]
        == pytest.approx(
            mlp_results["Mean F1"].max()
        )
    )

    # ------------------------------------------------------------------
    # Initial RF
    # ------------------------------------------------------------------

    grid_rf_initial = results[
        "grid_rf_initial"
    ]

    assert hasattr(
        grid_rf_initial,
        "cv_results_",
    )

    assert hasattr(
        grid_rf_initial,
        "best_estimator_",
    )

    assert grid_rf_initial.n_splits_ == 5

    # 2 × 3 × 2 = 12 RF configurations.
    assert len(
        grid_rf_initial.cv_results_["params"]
    ) == 12

    # ------------------------------------------------------------------
    # Improved RF
    # ------------------------------------------------------------------

    grid_rf_improved = results[
        "grid_rf_improved"
    ]

    assert hasattr(
        grid_rf_improved,
        "cv_results_",
    )

    assert hasattr(
        grid_rf_improved,
        "best_estimator_",
    )

    assert grid_rf_improved.n_splits_ == 5

    # 2 × 3 × 2 × 2 × 1 = 24 configurations.
    assert len(
        grid_rf_improved.cv_results_["params"]
    ) == 24

    # All model-selection scores must be valid Macro F1 values.
    for grid in [
        grid_lr,
        grid_rf_initial,
        grid_rf_improved,
    ]:
        scores = grid.cv_results_[
            "mean_test_score"
        ]

        assert np.isfinite(scores).all()
        assert np.all(
            (scores >= 0.0)
            & (scores <= 1.0)
        )


# ==============================================================================
# CBADVAI INITIAL TEST-SET EVALUATION
# ==============================================================================

def test_cbadvai_initial_models_test_set(
    cbadvai_model_selection,
):
    """
    Checkpoint 7:
    Evaluates the selected OLR, MLP, and initial RF models
    exactly once on the isolated test set.

    The test set is NOT used during model selection.
    """

    results = cbadvai_model_selection

    grid_lr = results["grid_lr"]
    grid_mlp = results["grid_mlp"]
    grid_rf_initial = results[
        "grid_rf_initial"
    ]

    # --------------------------------------------------------------
    # Predictions
    # --------------------------------------------------------------

    y_pred_lr = grid_lr.predict(X_test)
    y_pred_mlp = grid_mlp.predict(X_test)

    # RF uses expected-value prediction, matching the notebook.
    y_pred_rf = predict_ordinal_expected_value(
        grid_rf_initial,
        X_test,
    )

    predictions = {
        "Ordinal LR": y_pred_lr,
        "Optimized MLP": y_pred_mlp,
        "Initial RF": y_pred_rf,
    }

    # --------------------------------------------------------------
    # Evaluate each model
    # --------------------------------------------------------------

    for model_name, predictions_for_model in predictions.items():

        assert len(
            predictions_for_model
        ) == len(y_test)

        assert set(
            np.unique(predictions_for_model)
        ).issubset({1, 2, 3, 4, 5})

        metrics = evaluate_ordinal_model(
            y_test,
            predictions_for_model,
            model_name=model_name,
        )

        # Required output contract.
        assert "Macro_F1" in metrics
        assert "MAE" in metrics
        assert "QWK" in metrics
        assert "Confusion_Matrix" in metrics

        f1 = metrics["Macro_F1"]
        mae = metrics["MAE"]
        qwk = metrics["QWK"]
        cm = metrics["Confusion_Matrix"]

        # Mathematical bounds.
        assert 0.0 <= f1 <= 1.0
        assert 0.0 <= mae <= 4.0
        assert -1.0 <= qwk <= 1.0

        # Confusion matrix contract.
        assert isinstance(cm, np.ndarray)
        assert cm.shape == (5, 5)
        assert (cm >= 0).all()
        assert cm.sum() == len(y_test)

        # Established notebook performance gates.
        gate = INITIAL_MODEL_GATES[
            model_name
        ]

        assert f1 >= gate["min_f1"], (
            f"{model_name} Macro F1 degraded: "
            f"{f1:.4f} < {gate['min_f1']:.4f}"
        )

        assert mae <= gate["max_mae"], (
            f"{model_name} MAE degraded: "
            f"{mae:.4f} > {gate['max_mae']:.4f}"
        )


# ==============================================================================
# CBADVAI IMPROVED RF FINAL EVALUATION
# ==============================================================================

def test_cbadvai_improved_rf_final_evaluation(
    cbadvai_model_selection,
):
    """
    Checkpoint 8:
    Evaluates the improved RF on the isolated test set.

    Verifies:
        - valid predictions
        - Macro F1
        - MAE
        - QWK
        - confusion matrix
        - performance remains reasonably close to the notebook
        - improved RF is not materially worse than initial RF
    """

    results = cbadvai_model_selection

    grid_rf_initial = results[
        "grid_rf_initial"
    ]

    grid_rf_improved = results[
        "grid_rf_improved"
    ]

    # --------------------------------------------------------------
    # Initial RF
    # --------------------------------------------------------------

    initial_predictions = (
        predict_ordinal_expected_value(
            grid_rf_initial,
            X_test,
        )
    )

    initial_metrics = evaluate_ordinal_model(
        y_test,
        initial_predictions,
        model_name="Initial RF",
    )

    # --------------------------------------------------------------
    # Improved RF
    # --------------------------------------------------------------

    improved_predictions = (
        predict_ordinal_expected_value(
            grid_rf_improved,
            X_test,
        )
    )

    improved_metrics = evaluate_ordinal_model(
        y_test,
        improved_predictions,
        model_name="Improved RF",
    )

    # Prediction contract.
    assert len(improved_predictions) == len(y_test)

    assert set(
        np.unique(improved_predictions)
    ).issubset({1, 2, 3, 4, 5})

    # Metric contract.
    for key in [
        "Macro_F1",
        "MAE",
        "QWK",
        "Confusion_Matrix",
    ]:
        assert key in improved_metrics

    improved_f1 = improved_metrics["Macro_F1"]
    improved_mae = improved_metrics["MAE"]
    improved_qwk = improved_metrics["QWK"]

    initial_f1 = initial_metrics["Macro_F1"]
    initial_mae = initial_metrics["MAE"]

    # Mathematical bounds.
    assert 0.0 <= improved_f1 <= 1.0
    assert 0.0 <= improved_mae <= 4.0
    assert -1.0 <= improved_qwk <= 1.0

    # Confusion matrix.
    cm = improved_metrics["Confusion_Matrix"]

    assert isinstance(cm, np.ndarray)
    assert cm.shape == (5, 5)
    assert (cm >= 0).all()
    assert cm.sum() == len(y_test)

    # Established improved-RF performance gate.
    assert improved_f1 >= IMPROVED_RF_GATE[
        "min_f1"
    ]

    assert improved_mae <= IMPROVED_RF_GATE[
        "max_mae"
    ]

    assert improved_qwk >= IMPROVED_RF_GATE[
        "min_qwk"
    ]

    # --------------------------------------------------------------
    # Improvement / non-regression check.
    #
    # We allow a very small tolerance because model optimization
    # can produce numerically equivalent results with minor changes.
    # --------------------------------------------------------------

    assert improved_f1 >= initial_f1 - 0.03, (
        "Improved RF Macro F1 materially regressed "
        "relative to the initial RF."
    )

    assert improved_mae <= initial_mae + 0.05, (
        "Improved RF MAE materially regressed "
        "relative to the initial RF."
    )


# ==============================================================================
# CBADVAI FEATURE WEIGHT OUTPUT
# ==============================================================================

def test_cbadvai_feature_weight_output(
    cbadvai_model_selection,
):
    """
    Checkpoint 9:
    Verifies the final four-model feature-weight analysis.

    Models:
        - Ordinal LR
        - Optimized MLP
        - Initial RF
        - Improved RF

    The test validates the structure and mathematical normalization
    of the output and checks the established high-level feature
    interpretation without requiring exact notebook floating-point
    values.
    """

    results = cbadvai_model_selection

    weights_df = compute_summary_feature_weights(
        results["grid_lr"],
        results["grid_mlp"],
        results["grid_rf_initial"],
        results["grid_rf_improved"],
        X_test,
        y_test,
        feature_names=SELECTED_FEATURES,
    )

    # --------------------------------------------------------------
    # Output structure
    # --------------------------------------------------------------

    assert isinstance(
        weights_df,
        pd.DataFrame,
    )

    assert weights_df.shape == (
        11,
        4,
    )

    assert list(
        weights_df.index
    ) == EXPECTED_CBADVAI_FEATURES

    assert list(
        weights_df.columns
    ) == [
        "Ordinal LR",
        "Optimized MLP",
        "Initial RF",
        "Improved RF",
    ]

    # --------------------------------------------------------------
    # Numerical integrity
    # --------------------------------------------------------------

    assert not weights_df.isnull().any().any()

    assert np.isfinite(
        weights_df.to_numpy()
    ).all()

    assert (
        weights_df >= 0.0
    ).all().all()

    assert (
        weights_df <= 1.0
    ).all().all()

    # Every model's normalized weights must sum to 1.
    for column in weights_df.columns:
        assert weights_df[column].sum() == pytest.approx(
            1.0,
            abs=1e-4,
        )

    # --------------------------------------------------------------
    # Every model must actually assign some weight.
    # --------------------------------------------------------------

    for column in weights_df.columns:
        assert (
            weights_df[column].sum()
            > 0
        )

    # --------------------------------------------------------------
    # High-level notebook interpretation.
    #
    # We deliberately do NOT assert exact weight values.
    # We only verify that the established important features
    # remain represented among the strongest aggregate drivers.
    # --------------------------------------------------------------

    aggregate_weights = (
        weights_df.mean(axis=1)
    )

    top_features = set(
        aggregate_weights
        .nlargest(5)
        .index
    )

    established_features = {
        "Time_Friends",
        "Quality_Lecturer",
        "Time_SocicalMedia",
    }

    # At least two of the three established top drivers
    # should remain among the five strongest aggregate features.
    assert len(
        established_features.intersection(
            top_features
        )
    ) >= 2