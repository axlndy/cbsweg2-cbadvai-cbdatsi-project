# tests/test_regression.py
import os

import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats

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
    load_and_preprocess_data,
    get_train_test_split,
    SELECTED_FEATURES,
)

from src.cbadvai.models import (
    build_ordinal_lr_pipeline,
    build_mlp_pipeline,
    build_rf_pipeline,
)

from src.cbadvai.metrics import (
    run_full_evaluation,
    compute_summary_feature_weights,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

ACTUAL_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "Database paper.xlsx"
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


# Notebook-established initial test-set confusion matrices.
EXPECTED_INITIAL_CONFUSION = {
    "Dummy Baseline": np.array([
        [0, 0, 15, 0, 0],
        [0, 0, 22, 0, 0],
        [0, 0, 238, 0, 0],
        [0, 0, 138, 0, 0],
        [0, 0, 21, 0, 0],
    ]),

    "Ordinal LR": np.array([
        [5, 2, 1, 6, 1],
        [4, 11, 1, 4, 2],
        [43, 62, 73, 28, 32],
        [26, 21, 30, 31, 30],
        [3, 2, 4, 7, 5],
    ]),

    "Optimized MLP": np.array([
        [4, 3, 4, 4, 0],
        [4, 3, 7, 6, 2],
        [23, 23, 98, 60, 34],
        [17, 11, 45, 45, 20],
        [0, 3, 5, 4, 9],
    ]),

    "Random Forest": np.array([
        [1, 0, 12, 2, 0],
        [0, 3, 17, 2, 0],
        [3, 19, 148, 62, 6],
        [1, 14, 63, 52, 8],
        [0, 2, 7, 5, 7],
    ]),
}


# Notebook-established final Improved RF confusion matrix.
EXPECTED_FINAL_RF_CONFUSION = np.array([
    [1, 0, 12, 2, 0],
    [0, 3, 17, 2, 0],
    [3, 20, 153, 56, 6],
    [1, 12, 65, 52, 8],
    [0, 2, 7, 5, 7],
])


# Threshold-based regression gates.
#
# These intentionally use >= / <= rather than requiring exact
# floating-point metric values.
INITIAL_PERFORMANCE_GATES = {
    "Dummy Baseline": {
        "min_f1": 0.14,
        "max_mae": 0.55,
        "min_qwk": -0.01,
    },

    "Ordinal LR": {
        "min_f1": 0.22,
        "max_mae": 1.20,
        "min_qwk": 0.10,
    },

    "Optimized MLP": {
        "min_f1": 0.25,
        "max_mae": 0.98,
        "min_qwk": 0.13,
    },

    "Random Forest": {
        "min_f1": 0.30,
        "max_mae": 0.65,
        "min_qwk": 0.20,
    },
}


FINAL_RF_GATE = {
    "min_f1": 0.30,
    "max_mae": 0.65,
    "min_qwk": 0.20,
}


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def actual_cbdatsi_dataset(tmp_path_factory):
    """
    Reproduces the CBDATSI notebook data-ingestion and
    preprocessing path using the actual repository dataset.
    """

    if not os.path.exists(ACTUAL_DATA_PATH):
        pytest.fail(
            f"Actual dataset not found at: "
            f"{ACTUAL_DATA_PATH}"
        )

    cache_dir = tmp_path_factory.mktemp(
        "cbdatsi_regression_cache"
    )

    cache_path = os.path.join(
        cache_dir,
        "dataset_cache.pkl"
    )

    df = load_and_cache_dataset(
        ACTUAL_DATA_PATH,
        cache_path
    )

    assert validate_dataset(df) is True

    df = clean_and_typecast_data(df)
    df = perform_feature_engineering(df)

    return df


@pytest.fixture(scope="module")
def cbadvai_regression_artifacts(tmp_path_factory):
    """
    Reproduces the notebook's fixed 80/20 split and fits the
    notebook-established winning configurations.

    This regression fixture intentionally does NOT perform
    hyperparameter tuning. Model selection is tested separately
    by the system test.
    """

    if not os.path.exists(ACTUAL_DATA_PATH):
        pytest.fail(
            f"Actual dataset not found at: "
            f"{ACTUAL_DATA_PATH}"
        )

    cache_dir = tmp_path_factory.mktemp(
        "cbadvai_regression_cache"
    )

    cache_path = os.path.join(
        cache_dir,
        "dataset_cache.pkl"
    )

    X, y = load_and_preprocess_data(
        file_path=ACTUAL_DATA_PATH,
        use_cache=False,
        cache_path=cache_path,
    )

    X_train, X_test, y_train, y_test = (
        get_train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )
    )

    # --------------------------------------------------------
    # Notebook-established OLR configuration
    # l2 + C=0.01
    # --------------------------------------------------------

    lr = build_ordinal_lr_pipeline(
        random_state=42
    )

    lr.set_params(
        classifier__C=0.01,
        classifier__penalty="l2",
    )

    lr.fit(X_train, y_train)

    # --------------------------------------------------------
    # Notebook-established MLP configuration
    # Very Deep + tanh
    # --------------------------------------------------------

    mlp = build_mlp_pipeline(
        random_state=42
    )

    mlp.set_params(
        classifier__hidden_layer_sizes=(
            128,
            64,
            32,
            16,
        ),
        classifier__activation="tanh",
    )

    mlp.fit(X_train, y_train)

    # --------------------------------------------------------
    # Notebook-established Initial RF configuration
    # --------------------------------------------------------

    rf_initial = build_rf_pipeline(
        random_state=42
    )

    rf_initial.set_params(
        classifier__n_estimators=100,
        classifier__max_depth=20,
        classifier__min_samples_split=2,
    )

    rf_initial.fit(X_train, y_train)

    # --------------------------------------------------------
    # Notebook-established Improved RF configuration
    # --------------------------------------------------------

    rf_improved = build_rf_pipeline(
        random_state=42
    )

    rf_improved.set_params(
        classifier__n_estimators=300,
        classifier__max_depth=None,
        classifier__min_samples_split=2,
        classifier__min_samples_leaf=1,
        classifier__class_weight=None,
    )

    rf_improved.fit(X_train, y_train)

    # --------------------------------------------------------
    # Initial evaluation
    # --------------------------------------------------------

    models_initial, metrics_initial = (
        run_full_evaluation(
            X_train,
            y_train,
            X_test,
            y_test,
            lr,
            mlp,
            rf_initial,
            rf_label="Random Forest",
        )
    )

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    models_final, metrics_final = (
        run_full_evaluation(
            X_train,
            y_train,
            X_test,
            y_test,
            lr,
            mlp,
            rf_improved,
            rf_label="Improved RF",
        )
    )

    # --------------------------------------------------------
    # Feature weights
    # --------------------------------------------------------

    weights = compute_summary_feature_weights(
        grid_lr=lr,
        grid_mlp=mlp,
        grid_rf_initial=rf_initial,
        grid_rf_improved=rf_improved,
        X_test=X_test,
        y_test=y_test,
        feature_names=SELECTED_FEATURES,
    )

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "models_initial": models_initial,
        "metrics_initial": metrics_initial,
        "models_final": models_final,
        "metrics_final": metrics_final,
        "weights": weights,
    }


# ============================================================
# CBDATSI REGRESSION
# ============================================================

def test_regression_cbdatsi_data_integrity(
    actual_cbdatsi_dataset
):
    """
    Locks the notebook-established CBDATSI dataset state.

    Notebook baseline:
        rows = 2170
        missing values = 0
        duplicate rows = 0
    """

    assert len(actual_cbdatsi_dataset) == 2170

    assert (
        actual_cbdatsi_dataset.isnull().sum().sum()
        == 0
    )

    # The executed notebook reports zero duplicates after cleaning.
    assert (
        actual_cbdatsi_dataset.duplicated().sum()
        == 0
    )


def test_regression_cbdatsi_clustering_specification(
    actual_cbdatsi_dataset
):
    """Locks the 11-variable clustering specification."""

    features = get_clustering_features()

    assert len(features) == 11
    assert features == EXPECTED_CLUSTERING_FEATURES


def test_regression_cbdatsi_kmeans_silhouette(
    actual_cbdatsi_dataset
):
    """
    Locks the established K-Means configuration and
    silhouette baseline.
    """

    clustered_df, cluster_summary = (
        run_kmeans_clustering(
            actual_cbdatsi_dataset,
            n_clusters=3,
        )
    )

    assert len(clustered_df) == 2170

    assert (
        clustered_df["Cluster"].nunique()
        == 3
    )

    assert len(cluster_summary) == 3

    expected_summary_columns = (
        set(EXPECTED_CLUSTERING_FEATURES)
        | {"GPA"}
    )

    assert set(cluster_summary.columns) == (
        expected_summary_columns
    )

    score = evaluate_clusters(
        clustered_df
    )

    assert score == pytest.approx(
        0.1456,
        abs=1e-4,
    )


def test_regression_cbdatsi_chisquare_results(
    actual_cbdatsi_dataset
):
    """
    Locks the established clustering-to-Chi-Square output.
    """

    clustered_df, _ = (
        run_kmeans_clustering(
            actual_cbdatsi_dataset,
            n_clusters=3,
        )
    )

    chi2, p_val, dof, table = (
        perform_chisquare_independence(
            clustered_df,
            target_col="GPA",
            cluster_col="Cluster",
        )
    )

    assert chi2 == pytest.approx(
        43.2357,
        abs=1e-4,
    )

    assert dof == 8

    assert p_val == pytest.approx(
        7.9306e-7,
        rel=1e-4,
    )

    assert table.shape == (3, 5)
    assert table.values.sum() == 2170

    # Chi-Square assumption.
    _, _, _, expected_freq = (
        stats.chi2_contingency(table)
    )

    assert expected_freq.min() == pytest.approx(
        19.78,
        abs=1e-2,
    )

    assert expected_freq.min() >= 5

    # Exact observed contingency table.
    expected_table = pd.DataFrame(
        {
            1: [27, 27, 19],
            2: [50, 45, 14],
            3: [559, 335, 295],
            4: [300, 156, 236],
            5: [47, 25, 35],
        },
        index=pd.Index(
            [0, 1, 2],
            name="Cluster",
        ),
    )

    expected_table.columns.name = "GPA"

    pd.testing.assert_frame_equal(
        table,
        expected_table,
        check_index_type=False,
    )


# ============================================================
# CBADVAI DATA / SPLIT REGRESSION
# ============================================================

def test_regression_cbadvai_preprocessing_and_split(
    cbadvai_regression_artifacts
):
    """
    Locks the notebook's 11-feature preprocessing contract
    and 80/20 holdout split.
    """

    X = cbadvai_regression_artifacts["X"]
    y = cbadvai_regression_artifacts["y"]

    X_train = cbadvai_regression_artifacts[
        "X_train"
    ]

    X_test = cbadvai_regression_artifacts[
        "X_test"
    ]

    y_train = cbadvai_regression_artifacts[
        "y_train"
    ]

    y_test = cbadvai_regression_artifacts[
        "y_test"
    ]

    assert X.shape == (2170, 11)

    assert list(X.columns) == list(
        SELECTED_FEATURES
    )

    assert not X.isnull().any().any()

    assert set(
        y.unique()
    ).issubset({1, 2, 3, 4, 5})

    assert X_train.shape == (
        1736,
        11,
    )

    assert X_test.shape == (
        434,
        11,
    )

    assert len(y_train) == 1736
    assert len(y_test) == 434

    # Explicit holdout leakage protection.
    assert set(
        X_train.index
    ).isdisjoint(
        set(X_test.index)
    )

    assert set(
        y_train.index
    ).isdisjoint(
        set(y_test.index)
    )


# ============================================================
# CBADVAI PERFORMANCE REGRESSION
# ============================================================

def _metrics_to_dict(
    model_names,
    metrics_list
):
    return dict(
        zip(
            model_names,
            metrics_list
        )
    )


def _assert_metric_gate(
    metrics,
    model_name,
    gates
):
    result = metrics[model_name]

    assert (
        result["Macro_F1"]
        >= gates["min_f1"]
    )

    assert (
        result["MAE"]
        <= gates["max_mae"]
    )

    assert (
        result["QWK"]
        >= gates["min_qwk"]
    )

    confusion = result[
        "Confusion_Matrix"
    ]

    assert confusion.shape == (
        5,
        5,
    )

    assert confusion.sum() == 434

    assert (
        confusion >= 0
    ).all()


def test_regression_cbadvai_initial_model_performance(
    cbadvai_regression_artifacts
):
    """
    Protects the notebook's initial benchmark:

        Dummy
        Ordinal LR
        Optimized MLP
        Initial RF
    """

    metrics = _metrics_to_dict(
        cbadvai_regression_artifacts[
            "models_initial"
        ],
        cbadvai_regression_artifacts[
            "metrics_initial"
        ],
    )

    assert (
        cbadvai_regression_artifacts[
            "models_initial"
        ]
        == [
            "Dummy Baseline",
            "Ordinal LR",
            "Optimized MLP",
            "Random Forest",
        ]
    )

    for model_name, gates in (
        INITIAL_PERFORMANCE_GATES.items()
    ):
        _assert_metric_gate(
            metrics,
            model_name,
            gates,
        )

    # Established finding:
    # Initial RF had the highest Macro F1 among trained models.
    assert (
        metrics["Random Forest"]["Macro_F1"]
        >= metrics["Ordinal LR"]["Macro_F1"]
    )

    assert (
        metrics["Random Forest"]["Macro_F1"]
        >= metrics["Optimized MLP"]["Macro_F1"]
    )

    assert (
        metrics["Random Forest"]["Macro_F1"]
        >= metrics["Dummy Baseline"]["Macro_F1"]
    )

    # Protect the actual notebook confusion matrices.
    for (
        model_name,
        expected_matrix
    ) in EXPECTED_INITIAL_CONFUSION.items():

        np.testing.assert_array_equal(
            metrics[model_name][
                "Confusion_Matrix"
            ],
            expected_matrix,
        )


def test_regression_cbadvai_improved_rf_performance(
    cbadvai_regression_artifacts
):
    """
    Protects the notebook's final Improved RF result.
    """

    metrics = _metrics_to_dict(
        cbadvai_regression_artifacts[
            "models_final"
        ],
        cbadvai_regression_artifacts[
            "metrics_final"
        ],
    )

    assert (
        cbadvai_regression_artifacts[
            "models_final"
        ]
        == [
            "Dummy Baseline",
            "Ordinal LR",
            "Optimized MLP",
            "Improved RF",
        ]
    )

    _assert_metric_gate(
        metrics,
        "Improved RF",
        FINAL_RF_GATE,
    )

    improved = metrics[
        "Improved RF"
    ]

    # Improved RF must dominate the trained models.
    for model_name in [
        "Ordinal LR",
        "Optimized MLP",
    ]:

        assert (
            improved["Macro_F1"]
            >= metrics[model_name]["Macro_F1"]
        )

        assert (
            improved["MAE"]
            <= metrics[model_name]["MAE"]
        )

        assert (
            improved["QWK"]
            >= metrics[model_name]["QWK"]
        )

    initial = _metrics_to_dict(
        cbadvai_regression_artifacts[
            "models_initial"
        ],
        cbadvai_regression_artifacts[
            "metrics_initial"
        ],
    )["Random Forest"]

    # Improvement over initial RF.
    assert (
        improved["Macro_F1"]
        >= initial["Macro_F1"]
    )

    assert (
        improved["MAE"]
        <= initial["MAE"]
    )

    assert (
        improved["QWK"]
        >= initial["QWK"]
    )

    np.testing.assert_array_equal(
        improved["Confusion_Matrix"],
        EXPECTED_FINAL_RF_CONFUSION,
    )


# ============================================================
# CBADVAI FEATURE-WEIGHT REGRESSION
# ============================================================

def test_regression_cbadvai_feature_weight_summary(
    cbadvai_regression_artifacts
):
    """
    Protects the final four-model feature-weight mapping.

    The notebook's established top overall drivers are:
        Time_Friends
        Quality_Lecturer
        Time_SocicalMedia
    """

    weights = (
        cbadvai_regression_artifacts[
            "weights"
        ]
    )

    assert isinstance(
        weights,
        pd.DataFrame
    )

    assert weights.shape == (
        11,
        4,
    )

    assert list(weights.index) == list(
        SELECTED_FEATURES
    )

    assert list(weights.columns) == [
        "Ordinal LR",
        "Optimized MLP",
        "Initial RF",
        "Improved RF",
    ]

    assert not weights.isnull().any().any()

    assert np.isfinite(
        weights.to_numpy()
    ).all()

    assert (
        weights >= 0
    ).all().all()

    assert (
        weights <= 1
    ).all().all()

    # Every model's normalized weights must sum to 1.
    for column in weights.columns:
        assert (
            weights[column].sum()
            == pytest.approx(
                1.0,
                abs=1e-4,
            )
        )

    overall_mean = weights.mean(
        axis=1
    )

    assert (
        overall_mean.sum()
        == pytest.approx(
            1.0,
            abs=1e-4,
        )
    )

    top_three = set(
        overall_mean.nlargest(3).index
    )

    assert top_three == {
        "Time_Friends",
        "Quality_Lecturer",
        "Time_SocicalMedia",
    }