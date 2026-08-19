# tests/test_regression.py

import os

import numpy as np
import pandas as pd
import pytest

from src.cbdatsi.pipeline import (
    load_and_cache_dataset,
    validate_dataset,
    clean_and_typecast_data,
    perform_feature_engineering,
)

from src.cbdatsi.modeling import (
    get_clustering_features,
    run_kmeans_clustering,
    evaluate_clusters,
)

from src.cbdatsi.inference import (
    perform_chisquare_independence,
)

from src.cbadvai.preprocessing import (
    SELECTED_FEATURES,
    load_and_preprocess_data,
    get_train_test_split,
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


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

# -------------------------------------------------------------------------
# CBDATSI dataset
# -------------------------------------------------------------------------

ACTUAL_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "Database paper.xlsx",
)

EXPECTED_CLUSTERING_FEATURES = [
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

# -------------------------------------------------------------------------
# CBADVAI dataset/model expectations
# -------------------------------------------------------------------------

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

# -------------------------------------------------------------------------
# CBADVAI tolerant regression gates
#
# These intentionally DO NOT require exact notebook floating-point
# reproduction. They protect the established behavior of the analysis
# while allowing small numerical/model-selection differences.
# -------------------------------------------------------------------------

CBADVAI_INITIAL_REGRESSION_GATES = {
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

CBADVAI_IMPROVED_RF_REGRESSION_GATES = {
    "min_f1": 0.27,
    "max_mae": 0.75,
    "min_qwk": 0.18,
}


# =============================================================================
# CBDATSI FIXTURE
# =============================================================================

@pytest.fixture(scope="module")
def actual_dataset(tmp_path_factory):
    """
    Loads the actual CBDATSI dataset used by the notebook.

    A temporary cache is used so that regression tests always begin
    from the repository's actual Excel dataset rather than potentially
    loading a stale project cache.
    """

    if not os.path.exists(ACTUAL_DATA_PATH):
        pytest.fail(
            f"Actual CBDATSI dataset not found at: "
            f"{ACTUAL_DATA_PATH}"
        )

    cache_dir = tmp_path_factory.mktemp(
        "cbdatsi_regression_cache"
    )

    cache_path = os.path.join(
        cache_dir,
        "dataset_cache.pkl",
    )

    # -----------------------------------------------------------------
    # Reproduce the notebook preprocessing pipeline
    # -----------------------------------------------------------------

    df = load_and_cache_dataset(
        ACTUAL_DATA_PATH,
        cache_path,
    )

    validate_dataset(df)

    df = clean_and_typecast_data(df)

    df = perform_feature_engineering(df)

    return df


# =============================================================================
# CBDATSI DATASET REGRESSION
# =============================================================================

def test_regression_cbdatsi_data_integrity(
    actual_dataset,
):
    """
    Verifies that the actual CBDATSI dataset remains consistent
    with the documented notebook baseline.

    Established baseline:
        observations = 2170
        missing values = 0
        duplicate rows = 226
    """

    # Row count
    assert len(actual_dataset) == 2170, (
        "Dataset row count drifted from 2170."
    )

    # Missing values
    assert (
        actual_dataset.isnull().sum().sum()
        == 0
    ), (
        "Null values detected in the actual "
        "CBDATSI dataset."
    )

    # Duplicate rows
    #
    # This is intentionally kept as 226 because that is the
    # established dataset baseline used by the analysis.
    assert (
        actual_dataset.duplicated().sum()
        == 226
    ), (
        "Duplicate row count drifted from 226."
    )


# =============================================================================
# CBDATSI CLUSTERING SPECIFICATION REGRESSION
# =============================================================================

def test_regression_cbdatsi_clustering_specification(
    actual_dataset,
):
    """
    Verifies that the clustering model still uses exactly
    the 11 variables specified in the CBDATSI analysis.
    """

    features = get_clustering_features()

    assert len(features) == 11

    assert features == (
        EXPECTED_CLUSTERING_FEATURES
    )


# =============================================================================
# CBDATSI K-MEANS REGRESSION
# =============================================================================

def test_regression_cbdatsi_kmeans_silhouette(
    actual_dataset,
):
    """
    Verifies that the K-Means analysis still produces a valid
    clustering result.

    The notebook baseline silhouette was approximately 0.1456.

    Because floating-point/model-library changes can slightly
    affect the exact result, this test uses a tolerant range
    instead of requiring exact equality.
    """

    features = get_clustering_features()

    assert len(features) == 11

    clustered_df, cluster_summary = (
        run_kmeans_clustering(
            actual_dataset,
            n_clusters=3,
        )
    )

    # Observation conservation
    assert len(clustered_df) == 2170

    # Three clusters
    assert (
        clustered_df["Cluster"].nunique()
        == 3
    )

    # Cluster summary
    assert len(cluster_summary) == 3

    expected_summary_columns = (
        set(EXPECTED_CLUSTERING_FEATURES)
        | {"GPA"}
    )

    assert set(
        cluster_summary.columns
    ) == expected_summary_columns

    # Silhouette score
    score = evaluate_clusters(
        clustered_df
    )

    # Mathematical validity
    assert -1.0 <= score <= 1.0

    # Established notebook result was positive and
    # approximately 0.1456.
    #
    # We only require that the clustering remains
    # meaningfully positive rather than demanding
    # exact reproduction.
    assert score > 0.10, (
        f"K-Means silhouette score degraded unexpectedly: "
        f"{score:.4f}"
    )


# =============================================================================
# CBDATSI CHI-SQUARE REGRESSION
# =============================================================================

def test_regression_cbdatsi_chisquare_results(
    actual_dataset,
):
    """
    Verifies the complete CBDATSI clustering-to-inference step.

    The regression contract focuses on:
        - valid Chi-Square statistic
        - valid p-value
        - valid degrees of freedom
        - valid contingency-table structure
        - conservation of observations
        - statistically significant association
    """

    clustered_df, _ = run_kmeans_clustering(
        actual_dataset,
        n_clusters=3,
    )

    chi2, p_value, dof, contingency_table = (
        perform_chisquare_independence(
            clustered_df,
            target_col="GPA",
            cluster_col="Cluster",
        )
    )

    # Chi-square statistic cannot be negative.
    assert chi2 >= 0.0

    # p-value must be within [0, 1].
    assert 0.0 <= p_value <= 1.0

    # Degrees of freedom must be non-negative.
    assert dof >= 0

    # Three clusters × five GPA categories.
    assert contingency_table.shape == (
        3,
        5,
    )

    # Contingency table must conserve all observations.
    assert (
        contingency_table.values.sum()
        == len(clustered_df)
    )

    # Established analysis found a statistically significant
    # association between cluster membership and GPA.
    assert p_value < 0.05, (
        f"Chi-Square association is no longer statistically "
        f"significant: p={p_value:.6f}"
    )


# =============================================================================
# CBADVAI PREPROCESSING REGRESSION
# =============================================================================

def test_regression_cbadvai_preprocessing():
    """
    Verifies the CBADVAI preprocessing contract.

    Checks:
        - exact 11-feature selection
        - no target leakage
        - no missing values
        - valid GPA classes
        - 80/20 stratified split
        - train/test separation
    """

    X, y = load_and_preprocess_data(
        file_path=ACTUAL_DATA_PATH,
        use_cache=False,
    )

    # -----------------------------------------------------------------
    # Feature contract
    # -----------------------------------------------------------------

    assert len(SELECTED_FEATURES) == 11

    assert list(
        SELECTED_FEATURES
    ) == EXPECTED_CBADVAI_FEATURES

    assert X.shape[1] == 11

    assert list(
        X.columns
    ) == EXPECTED_CBADVAI_FEATURES

    # -----------------------------------------------------------------
    # Missing values
    # -----------------------------------------------------------------

    assert not X.isnull().any().any(), (
        "CBADVAI features contain missing values."
    )

    assert not y.isnull().any(), (
        "CBADVAI target contains missing values."
    )

    # -----------------------------------------------------------------
    # Target integrity
    # -----------------------------------------------------------------

    assert len(X) == len(y)

    assert set(
        np.unique(y)
    ).issubset(
        {1, 2, 3, 4, 5}
    )

    # GPA must not be included among the features.
    assert "GPA" not in X.columns

    # -----------------------------------------------------------------
    # Stratified 80/20 split
    # -----------------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        get_train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )
    )

    assert (
        len(X_train) + len(X_test)
        == len(X)
    )

    assert (
        len(y_train) + len(y_test)
        == len(y)
    )

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    # Approximately 80/20.
    assert (
        len(X_train) / len(X)
        == pytest.approx(
            0.80,
            abs=0.02,
        )
    )

    assert (
        len(X_test) / len(X)
        == pytest.approx(
            0.20,
            abs=0.02,
        )
    )

    # Every GPA class remains represented.
    assert set(
        np.unique(y_train)
    ) == {1, 2, 3, 4, 5}

    assert set(
        np.unique(y_test)
    ) == {1, 2, 3, 4, 5}

    # Explicit train/test index separation.
    train_indices = set(
        X_train.index
    )

    test_indices = set(
        X_test.index
    )

    assert train_indices.isdisjoint(
        test_indices
    ), (
        "Training and test observations overlap."
    )


# =============================================================================
# CBADVAI 5-FOLD MODEL SELECTION REGRESSION
# =============================================================================

def test_regression_cbadvai_five_fold_model_selection():
    """
    Verifies that the actual CBADVAI model-selection pipeline
    performs genuine 5-fold cross-validation.

    This test deliberately exercises the real tuning functions
    rather than recreating their logic inside the test.

    Exact notebook scores are NOT required.
    """

    from sklearn.model_selection import (
        StratifiedKFold,
    )

    X, y = load_and_preprocess_data(
        file_path=ACTUAL_DATA_PATH,
        use_cache=False,
    )

    X_train, X_test, y_train, y_test = (
        get_train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # -----------------------------------------------------------------
    # OLR
    # -----------------------------------------------------------------

    grid_lr = tune_ordinal_lr(
        X_train,
        y_train,
        cv=cv,
    )

    assert hasattr(
        grid_lr,
        "cv_results_",
    )

    assert hasattr(
        grid_lr,
        "best_estimator_",
    )

    assert grid_lr.n_splits_ == 5

    # C = 4 values × penalty = 2 values
    assert len(
        grid_lr.cv_results_["params"]
    ) == 8

    lr_scores = grid_lr.cv_results_[
        "mean_test_score"
    ]

    assert np.isfinite(
        lr_scores
    ).all()

    assert (
        (lr_scores >= 0.0)
        & (lr_scores <= 1.0)
    ).all()

    # -----------------------------------------------------------------
    # MLP
    # -----------------------------------------------------------------

    grid_mlp, mlp_results, architecture_keys, best_mlp = (
        tune_mlp(
            X_train,
            y_train,
            cv=cv,
        )
    )

    assert isinstance(
        mlp_results,
        pd.DataFrame,
    )

    # 7 architectures × 3 activations
    assert len(mlp_results) == 21

    assert set(
        mlp_results["Activation"]
    ) == {
        "relu",
        "tanh",
        "logistic",
    }

    assert mlp_results[
        "Mean F1"
    ].notna().all()

    assert np.isfinite(
        mlp_results["Mean F1"]
    ).all()

    assert (
        mlp_results["Mean F1"]
        .between(0.0, 1.0)
        .all()
    )

    assert (
        best_mlp["Mean F1"]
        == pytest.approx(
            mlp_results[
                "Mean F1"
            ].max()
        )
    )

    # Returned model must be fitted.
    assert hasattr(
        grid_mlp,
        "predict",
    )

    # -----------------------------------------------------------------
    # Initial RF
    # -----------------------------------------------------------------

    grid_rf_initial = tune_initial_rf(
        X_train,
        y_train,
        cv=cv,
    )

    assert hasattr(
        grid_rf_initial,
        "cv_results_",
    )

    assert hasattr(
        grid_rf_initial,
        "best_estimator_",
    )

    assert grid_rf_initial.n_splits_ == 5

    # 2 estimators × 3 depths × 2 split settings
    assert len(
        grid_rf_initial.cv_results_[
            "params"
        ]
    ) == 12

    rf_initial_scores = (
        grid_rf_initial.cv_results_[
            "mean_test_score"
        ]
    )

    assert np.isfinite(
        rf_initial_scores
    ).all()

    assert (
        (rf_initial_scores >= 0.0)
        & (rf_initial_scores <= 1.0)
    ).all()

    # -----------------------------------------------------------------
    # Improved RF
    # -----------------------------------------------------------------

    grid_rf_improved = tune_improved_rf(
        X_train,
        y_train,
        cv=cv,
    )

    assert hasattr(
        grid_rf_improved,
        "cv_results_",
    )

    assert hasattr(
        grid_rf_improved,
        "best_estimator_",
    )

    assert grid_rf_improved.n_splits_ == 5

    improved_scores = (
        grid_rf_improved.cv_results_[
            "mean_test_score"
        ]
    )

    assert np.isfinite(
        improved_scores
    ).all()

    assert (
        (improved_scores >= 0.0)
        & (improved_scores <= 1.0)
    ).all()


# =============================================================================
# CBADVAI INITIAL MODEL TEST-SET REGRESSION
# =============================================================================

def test_regression_cbadvai_initial_models_test_set():
    """
    Verifies the established CBADVAI initial-model performance
    on the isolated test set.

    Models:
        - Ordinal LR
        - Optimized MLP
        - Initial RF

    The test intentionally uses tolerant performance gates rather
    than exact notebook floating-point values.
    """

    from sklearn.model_selection import (
        StratifiedKFold,
    )

    X, y = load_and_preprocess_data(
        file_path=ACTUAL_DATA_PATH,
        use_cache=False,
    )

    X_train, X_test, y_train, y_test = (
        get_train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # -----------------------------------------------------------------
    # Fit selected models
    # -----------------------------------------------------------------

    grid_lr = tune_ordinal_lr(
        X_train,
        y_train,
        cv=cv,
    )

    grid_mlp, _, _, _ = tune_mlp(
        X_train,
        y_train,
        cv=cv,
    )

    grid_rf_initial = tune_initial_rf(
        X_train,
        y_train,
        cv=cv,
    )

    # -----------------------------------------------------------------
    # Test-set predictions
    # -----------------------------------------------------------------

    predictions = {
        "Ordinal LR": grid_lr.predict(
            X_test
        ),

        "Optimized MLP": grid_mlp.predict(
            X_test
        ),

        "Initial RF": predict_ordinal_expected_value(
            grid_rf_initial,
            X_test,
        ),
    }

    # -----------------------------------------------------------------
    # Evaluate models
    # -----------------------------------------------------------------

    for model_name, y_pred in predictions.items():

        assert len(y_pred) == len(y_test)

        assert set(
            np.unique(y_pred)
        ).issubset(
            {1, 2, 3, 4, 5}
        )

        metrics = evaluate_ordinal_model(
            y_test,
            y_pred,
            model_name=model_name,
        )

        assert "Macro_F1" in metrics
        assert "MAE" in metrics
        assert "QWK" in metrics
        assert "Confusion_Matrix" in metrics

        f1 = metrics["Macro_F1"]
        mae = metrics["MAE"]
        qwk = metrics["QWK"]
        confusion_matrix = metrics[
            "Confusion_Matrix"
        ]

        # Mathematical bounds.
        assert 0.0 <= f1 <= 1.0
        assert 0.0 <= mae <= 4.0
        assert -1.0 <= qwk <= 1.0

        # Confusion matrix.
        assert isinstance(
            confusion_matrix,
            np.ndarray,
        )

        assert confusion_matrix.shape == (
            5,
            5,
        )

        assert (
            confusion_matrix >= 0
        ).all()

        assert (
            confusion_matrix.sum()
            == len(y_test)
        )

        # Tolerant regression gates.
        gate = (
            CBADVAI_INITIAL_REGRESSION_GATES[
                model_name
            ]
        )

        assert f1 >= gate["min_f1"], (
            f"{model_name} Macro F1 degraded: "
            f"{f1:.4f} < "
            f"{gate['min_f1']:.4f}"
        )

        assert mae <= gate["max_mae"], (
            f"{model_name} MAE degraded: "
            f"{mae:.4f} > "
            f"{gate['max_mae']:.4f}"
        )


# =============================================================================
# CBADVAI IMPROVED RANDOM FOREST REGRESSION
# =============================================================================

def test_regression_cbadvai_improved_rf_final_evaluation():
    """
    Verifies the final improved Random Forest evaluation.

    Established notebook result was approximately:
        Macro F1 = 0.3130
        MAE      = 0.6221
        QWK      = 0.2257

    Exact equality is intentionally avoided.
    """

    from sklearn.model_selection import (
        StratifiedKFold,
    )

    X, y = load_and_preprocess_data(
        file_path=ACTUAL_DATA_PATH,
        use_cache=False,
    )

    X_train, X_test, y_train, y_test = (
        get_train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # -----------------------------------------------------------------
    # Initial RF
    # -----------------------------------------------------------------

    grid_rf_initial = tune_initial_rf(
        X_train,
        y_train,
        cv=cv,
    )

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

    # -----------------------------------------------------------------
    # Improved RF
    # -----------------------------------------------------------------

    grid_rf_improved = tune_improved_rf(
        X_train,
        y_train,
        cv=cv,
    )

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

    # -----------------------------------------------------------------
    # Prediction contract
    # -----------------------------------------------------------------

    assert len(
        improved_predictions
    ) == len(y_test)

    assert set(
        np.unique(improved_predictions)
    ).issubset(
        {1, 2, 3, 4, 5}
    )

    # -----------------------------------------------------------------
    # Metric contract
    # -----------------------------------------------------------------

    for metric_name in [
        "Macro_F1",
        "MAE",
        "QWK",
        "Confusion_Matrix",
    ]:
        assert metric_name in (
            improved_metrics
        )

    improved_f1 = (
        improved_metrics["Macro_F1"]
    )

    improved_mae = (
        improved_metrics["MAE"]
    )

    improved_qwk = (
        improved_metrics["QWK"]
    )

    initial_f1 = (
        initial_metrics["Macro_F1"]
    )

    initial_mae = (
        initial_metrics["MAE"]
    )

    # Mathematical bounds.
    assert 0.0 <= improved_f1 <= 1.0
    assert 0.0 <= improved_mae <= 4.0
    assert -1.0 <= improved_qwk <= 1.0

    # -----------------------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------------------

    confusion_matrix = (
        improved_metrics[
            "Confusion_Matrix"
        ]
    )

    assert isinstance(
        confusion_matrix,
        np.ndarray,
    )

    assert confusion_matrix.shape == (
        5,
        5,
    )

    assert (
        confusion_matrix >= 0
    ).all()

    assert (
        confusion_matrix.sum()
        == len(y_test)
    )

    # -----------------------------------------------------------------
    # Tolerant notebook regression gates
    # -----------------------------------------------------------------

    assert (
        improved_f1
        >= CBADVAI_IMPROVED_RF_REGRESSION_GATES[
            "min_f1"
        ]
    ), (
        f"Improved RF Macro F1 degraded: "
        f"{improved_f1:.4f}"
    )

    assert (
        improved_mae
        <= CBADVAI_IMPROVED_RF_REGRESSION_GATES[
            "max_mae"
        ]
    ), (
        f"Improved RF MAE degraded: "
        f"{improved_mae:.4f}"
    )

    assert (
        improved_qwk
        >= CBADVAI_IMPROVED_RF_REGRESSION_GATES[
            "min_qwk"
        ]
    ), (
        f"Improved RF QWK degraded: "
        f"{improved_qwk:.4f}"
    )

    # -----------------------------------------------------------------
    # Non-regression relative to initial RF
    #
    # Small tolerance is allowed because model-selection and
    # library-version differences can produce small numerical
    # changes.
    # -----------------------------------------------------------------

    assert improved_f1 >= (
        initial_f1 - 0.03
    ), (
        "Improved RF Macro F1 materially "
        "regressed relative to Initial RF."
    )

    assert improved_mae <= (
        initial_mae + 0.05
    ), (
        "Improved RF MAE materially "
        "regressed relative to Initial RF."
    )


# =============================================================================
# CBADVAI FEATURE WEIGHT REGRESSION
# =============================================================================

def test_regression_cbadvai_feature_weight_summary():
    """
    Verifies the final four-model CBADVAI feature-weight analysis.

    Models:
        - Ordinal LR
        - Optimized MLP
        - Initial RF
        - Improved RF

    The regression test protects:
        - 11-feature structure
        - 4-model structure
        - normalized weights
        - finite/non-negative values
        - high-level notebook interpretation

    Exact floating-point feature weights are intentionally NOT
    required to match the notebook exactly.
    """

    from sklearn.model_selection import (
        StratifiedKFold,
    )

    X, y = load_and_preprocess_data(
        file_path=ACTUAL_DATA_PATH,
        use_cache=False,
    )

    X_train, X_test, y_train, y_test = (
        get_train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # -----------------------------------------------------------------
    # Train all four models
    # -----------------------------------------------------------------

    grid_lr = tune_ordinal_lr(
        X_train,
        y_train,
        cv=cv,
    )

    grid_mlp, _, _, _ = tune_mlp(
        X_train,
        y_train,
        cv=cv,
    )

    grid_rf_initial = tune_initial_rf(
        X_train,
        y_train,
        cv=cv,
    )

    grid_rf_improved = tune_improved_rf(
        X_train,
        y_train,
        cv=cv,
    )

    # -----------------------------------------------------------------
    # Compute feature weights
    # -----------------------------------------------------------------

    weights_df = (
        compute_summary_feature_weights(
            grid_lr,
            grid_mlp,
            grid_rf_initial,
            grid_rf_improved,
            X_test,
            y_test,
            feature_names=SELECTED_FEATURES,
        )
    )

    # -----------------------------------------------------------------
    # Output structure
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Numerical integrity
    # -----------------------------------------------------------------

    assert not (
        weights_df.isnull().any().any()
    )

    assert np.isfinite(
        weights_df.to_numpy()
    ).all()

    assert (
        weights_df >= 0.0
    ).all().all()

    assert (
        weights_df <= 1.0
    ).all().all()

    # Every model's feature weights are normalized.
    for column in weights_df.columns:

        assert (
            weights_df[column].sum()
            == pytest.approx(
                1.0,
                abs=1e-4,
            )
        )

    # -----------------------------------------------------------------
    # High-level interpretation
    #
    # We do NOT require exact feature weights because these can vary
    # slightly across environments/model versions.
    # -----------------------------------------------------------------

    aggregate_weights = (
        weights_df.mean(axis=1)
    )

    top_five_features = set(
        aggregate_weights
        .nlargest(5)
        .index
    )

    # Established analysis emphasized these variables.
    established_features = {
        "Time_Friends",
        "Quality_Lecturer",
        "Time_SocicalMedia",
    }

    # At least two of the established important features should
    # remain among the five strongest aggregate drivers.
    assert len(
        established_features.intersection(
            top_five_features
        )
    ) >= 2, (
        "Feature-weight interpretation has materially "
        "changed from the established notebook result."
    )