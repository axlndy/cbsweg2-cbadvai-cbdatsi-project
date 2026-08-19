# tests/test_system_overall.py
"""
End-to-End System Test for the integrated CBDATSI + CBADVAI pipeline.

CBDATSI:
Load Dataset
    -> Validation
    -> Cleaning
    -> Feature Engineering
    -> EDA
    -> K-Means
    -> Silhouette Evaluation
    -> Chi-Square Inference
    -> Final Statistical Output

CBADVAI:
Load Dataset
    -> Preprocessing
    -> 80/20 Holdout
    -> Leakage Checks
    -> Actual 5-Fold Model Selection
    -> Initial Test Evaluation
    -> Improved RF Selection
    -> Final Test Evaluation
    -> Feature Weight Output
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from unittest.mock import patch

from sklearn.model_selection import StratifiedKFold
from scipy.stats import chi2_contingency

from src.cbdatsi.pipeline import (
    load_and_cache_dataset,
    validate_dataset,
    clean_and_typecast_data,
    perform_feature_engineering,
)

from src.cbdatsi.eda_plots import (
    plot_demographics,
    plot_behavioral_boxplots,
    plot_socioeconomic_conditional,
    plot_institutional_heatmap,
    plot_mindset_heatmaps,
)

from src.cbdatsi.modeling import (
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
    tune_ordinal_lr,
    tune_mlp,
    tune_initial_rf,
    tune_improved_rf,
)

from src.cbadvai.metrics import (
    run_full_evaluation,
    compute_summary_feature_weights,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RAW_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "Database paper.xlsx",
)


EXPECTED_FEATURES = [
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


# Thresholds are based on the notebook's established results.
# They intentionally use >= / <= instead of exact floating-point equality.

INITIAL_GATES = {
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
# SYSTEM DATA FIXTURE
# ============================================================

@pytest.fixture(scope="module")
def system_data(tmp_path_factory):
    """
    Creates the actual system states used throughout the
    end-to-end tests.
    """

    if not os.path.exists(
        RAW_DATA_PATH
    ):
        pytest.fail(
            f"Actual dataset not found at: "
            f"{RAW_DATA_PATH}"
        )

    cache_dir = (
        tmp_path_factory.mktemp(
            "system_test_cache"
        )
    )

    cbdatsi_cache = os.path.join(
        cache_dir,
        "cbdatsi_cache.pkl"
    )

    cbadvai_cache = os.path.join(
        cache_dir,
        "cbadvai_cache.pkl"
    )

    # --------------------------------------------------------
    # CBDATSI
    # --------------------------------------------------------

    df_raw = load_and_cache_dataset(
        raw_path=RAW_DATA_PATH,
        cache_path=cbdatsi_cache,
    )

    assert validate_dataset(
        df_raw
    ) is True

    df_cleaned = (
        clean_and_typecast_data(
            df_raw
        )
    )

    df_engineered = (
        perform_feature_engineering(
            df_cleaned
        )
    )

    # --------------------------------------------------------
    # CBADVAI
    # --------------------------------------------------------

    X, y = load_and_preprocess_data(
        file_path=RAW_DATA_PATH,
        use_cache=False,
        cache_path=cbadvai_cache,
    )

    X_train, X_test, y_train, y_test = (
        get_train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )
    )

    return {
        "df_raw": df_raw,
        "df_cleaned": df_cleaned,
        "df_engineered": df_engineered,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


# ============================================================
# CBDATSI: LOAD -> PREPROCESSING
# ============================================================

def test_cbdatsi_load_and_preprocessing(
    system_data
):
    """
    Checkpoint 1:
    Load Dataset -> Validation -> Cleaning -> Feature Engineering.
    """

    df_raw = system_data[
        "df_raw"
    ]

    df_cleaned = system_data[
        "df_cleaned"
    ]

    df_engineered = system_data[
        "df_engineered"
    ]

    assert df_raw.shape == (
        2170,
        22,
    )

    assert validate_dataset(
        df_raw
    ) is True

    assert (
        df_raw.isnull().sum().sum()
        == 0
    )

    # Notebook-established cleaning result.
    assert (
        df_cleaned.duplicated().sum()
        == 0
    )

    required_labels = [
        "GPA_Label",
        "Year_Label",
        "Gender_Label",
        "Poor_Stu_Label",
        "Policy_Stu_Label",
    ]

    for label in required_labels:
        assert (
            label in df_engineered.columns
        )

        assert (
            df_engineered[label]
            .notna()
            .all()
        )

    assert len(
        df_engineered
    ) == len(df_raw)


# ============================================================
# CBDATSI: EDA INTEGRATION
# ============================================================

@patch(
    "matplotlib.pyplot.show"
)
def test_cbdatsi_eda_integration(
    mock_show,
    system_data
):
    """
    Checkpoint 2:

    Feature-engineered data
        -> all five EDA plotting functions
        -> figures successfully generated
        -> pipeline remains usable.
    """

    df = system_data[
        "df_engineered"
    ]

    eda_functions = [
        plot_demographics,
        plot_behavioral_boxplots,
        plot_socioeconomic_conditional,
        plot_institutional_heatmap,
        plot_mindset_heatmaps,
    ]

    for plot_function in (
        eda_functions
    ):

        plt.close("all")

        plot_function(df)

        assert (
            len(
                plt.get_fignums()
            )
            >= 1
        ), (
            f"{plot_function.__name__} "
            "did not create a figure."
        )

    assert (
        mock_show.call_count
        == len(eda_functions)
    )

    plt.close("all")


# ============================================================
# CBDATSI: CLUSTERING
# ============================================================

def test_cbdatsi_clustering_integration(
    system_data
):
    """
    Checkpoint 3:

    Feature-engineered / EDA-ready data
        -> K-Means
        -> cluster labels
        -> cluster summary
        -> silhouette evaluation.
    """

    df_engineered = (
        system_data[
            "df_engineered"
        ]
    )

    df_raw = system_data[
        "df_raw"
    ]

    df_clustered, cluster_summary = (
        run_kmeans_clustering(
            df_engineered,
            n_clusters=3,
        )
    )

    assert len(
        df_clustered
    ) == len(df_raw)

    assert (
        "Cluster"
        in df_clustered.columns
    )

    assert set(
        df_clustered[
            "Cluster"
        ].unique()
    ).issubset({
        0,
        1,
        2,
    })

    assert (
        df_clustered[
            "Cluster"
        ].nunique()
        == 3
    )

    assert len(
        cluster_summary
    ) == 3

    assert set(
        cluster_summary.columns
    ) == (
        set(EXPECTED_FEATURES)
        | {"GPA"}
    )

    silhouette = (
        evaluate_clusters(
            df_clustered
        )
    )

    assert -1.0 <= silhouette <= 1.0

    assert silhouette >= 0.10


# ============================================================
# CBDATSI: INFERENCE -> OUTPUT
# ============================================================

def test_cbdatsi_inference_and_output(
    system_data
):
    """
    Checkpoint 4:

    Clustered data
        -> Chi-Square inference
        -> statistical output
        -> assumption verification.
    """

    df_engineered = (
        system_data[
            "df_engineered"
        ]
    )

    df_raw = system_data[
        "df_raw"
    ]

    df_clustered, _ = (
        run_kmeans_clustering(
            df_engineered,
            n_clusters=3,
        )
    )

    chi2, p_val, dof, table = (
        perform_chisquare_independence(
            df_clustered,
            target_col="GPA",
            cluster_col="Cluster",
        )
    )

    assert table.shape == (
        3,
        5,
    )

    assert (
        table.values.sum()
        == len(df_raw)
    )

    assert chi2 >= 0.0

    assert (
        0.0 <= p_val <= 1.0
    )

    assert p_val < 0.05

    assert dof == 8

    # Explicit Chi-Square expected-frequency assumption.
    _, _, _, expected_freq = (
        chi2_contingency(table)
    )

    assert np.isfinite(
        expected_freq
    ).all()

    assert (
        expected_freq.min()
        >= 5
    )


# ============================================================
# CBADVAI: PREPROCESSING -> SPLIT
# ============================================================

def test_cbadvai_preprocessing_and_split(
    system_data
):
    """
    Checkpoint 5:

    Load Dataset
        -> 11-feature preprocessing
        -> 80/20 holdout
        -> class preservation
        -> leakage checks.
    """

    X = system_data["X"]
    y = system_data["y"]

    X_train = system_data[
        "X_train"
    ]

    X_test = system_data[
        "X_test"
    ]

    y_train = system_data[
        "y_train"
    ]

    y_test = system_data[
        "y_test"
    ]

    assert X.shape == (
        2170,
        11,
    )

    assert list(
        X.columns
    ) == list(
        SELECTED_FEATURES
    )

    assert list(
        X.columns
    ) == EXPECTED_FEATURES

    assert not X.isnull().any().any()

    assert set(
        np.unique(y)
    ).issubset({
        1,
        2,
        3,
        4,
        5,
    })

    assert X_train.shape == (
        1736,
        11,
    )

    assert X_test.shape == (
        434,
        11,
    )

    assert len(
        y_train
    ) == 1736

    assert len(
        y_test
    ) == 434

    # --------------------------------------------------------
    # Holdout leakage
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Class coverage
    # --------------------------------------------------------

    assert set(
        np.unique(y_train)
    ) == {
        1,
        2,
        3,
        4,
        5,
    }

    assert set(
        np.unique(y_test)
    ) == {
        1,
        2,
        3,
        4,
        5,
    }

    # --------------------------------------------------------
    # Stratification
    # --------------------------------------------------------

    train_props = (
        y_train
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    full_props = (
        y
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    np.testing.assert_allclose(
        train_props.values,
        full_props.values,
        atol=0.03,
    )


# ============================================================
# CBADVAI: ACTUAL 5-FOLD MODEL SELECTION
# ============================================================

@pytest.fixture(scope="module")
def model_selection_artifacts(
    system_data
):
    """
    Executes the REAL model-selection pipeline once.

    This is deliberately based on the same tuning functions used
    by the CBADVAI notebook.
    """

    X_train = system_data[
        "X_train"
    ]

    y_train = system_data[
        "y_train"
    ]

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    grid_lr = tune_ordinal_lr(
        X_train,
        y_train,
        skf,
    )

    (
        grid_mlp,
        mlp_results,
        arch_keys,
        best_mlp_row,
    ) = tune_mlp(
        X_train,
        y_train,
        skf,
    )

    grid_rf_initial = (
        tune_initial_rf(
            X_train,
            y_train,
            skf,
        )
    )

    grid_rf_improved = (
        tune_improved_rf(
            X_train,
            y_train,
            skf,
        )
    )

    splits = list(
        skf.split(
            X_train,
            y_train,
        )
    )

    return {
        "skf": skf,
        "splits": splits,
        "grid_lr": grid_lr,
        "grid_mlp": grid_mlp,
        "mlp_results": mlp_results,
        "arch_keys": arch_keys,
        "best_mlp_row": best_mlp_row,
        "grid_rf_initial": grid_rf_initial,
        "grid_rf_improved": grid_rf_improved,
    }


def test_cbadvai_actual_five_fold_model_selection(
    system_data,
    model_selection_artifacts
):
    """
    Checkpoint 6:

    Verifies that the ACTUAL model-selection pipeline
    successfully performs five-fold CV.

    This is not merely a test that StratifiedKFold can be
    instantiated. The actual tuning functions have already
    been executed by the fixture.
    """

    X_train = system_data[
        "X_train"
    ]

    y_train = system_data[
        "y_train"
    ]

    artifacts = (
        model_selection_artifacts
    )

    splits = artifacts[
        "splits"
    ]

    # --------------------------------------------------------
    # Five-fold integrity
    # --------------------------------------------------------

    assert len(
        splits
    ) == 5

    all_validation_indices = []

    for train_idx, val_idx in splits:

        assert set(
            train_idx
        ).isdisjoint(
            set(val_idx)
        )

        assert (
            len(train_idx)
            + len(val_idx)
            == len(X_train)
        )

        assert set(
            np.unique(
                y_train.iloc[
                    val_idx
                ]
            )
        ) == {
            1,
            2,
            3,
            4,
            5,
        }

        all_validation_indices.extend(
            val_idx.tolist()
        )

    # Every training observation is used exactly once
    # as validation data across the five folds.
    assert (
        len(all_validation_indices)
        == len(X_train)
    )

    assert (
        len(
            set(
                all_validation_indices
            )
        )
        == len(X_train)
    )

    # --------------------------------------------------------
    # GridSearchCV verification
    # --------------------------------------------------------

    for grid in [
        artifacts["grid_lr"],
        artifacts["grid_rf_initial"],
        artifacts["grid_rf_improved"],
    ]:

        assert (
            grid.n_splits_
            == 5
        )

        for fold in range(5):

            assert (
                f"split{fold}_test_score"
                in grid.cv_results_
            )

    # --------------------------------------------------------
    # MLP five-fold verification
    # --------------------------------------------------------

    mlp_results = (
        artifacts[
            "mlp_results"
        ]
    )

    assert len(
        mlp_results
    ) == 21

    fold_columns = [
        f"Fold_{i}_F1"
        for i in range(1, 6)
    ]

    assert all(
        column in mlp_results.columns
        for column in fold_columns
    )

    assert not (
        mlp_results[
            fold_columns
        ]
        .isnull()
        .any()
        .any()
    )

    for _, row in (
        mlp_results.iterrows()
    ):

        fold_mean = (
            row[
                fold_columns
            ]
            .astype(float)
            .mean()
        )

        assert (
            row["Mean F1"]
            == pytest.approx(
                fold_mean,
                abs=1e-12,
            )
        )

    # --------------------------------------------------------
    # Established winning configurations
    # --------------------------------------------------------

    grid_lr = (
        artifacts["grid_lr"]
    )

    assert (
        grid_lr.best_params_[
            "classifier__C"
        ]
        == 0.01
    )

    assert (
        grid_lr.best_params_[
            "classifier__penalty"
        ]
        == "l2"
    )

    assert (
        grid_lr.best_score_
        >= 0.22
    )

    # MLP.
    best_mlp_row = (
        artifacts[
            "best_mlp_row"
        ]
    )

    assert (
        best_mlp_row[
            "Architecture"
        ]
        == "Very Deep (128, 64, 32, 16)"
    )

    assert (
        best_mlp_row[
            "Activation"
        ]
        == "tanh"
    )

    assert (
        best_mlp_row[
            "Mean F1"
        ]
        >= 0.26
    )

    grid_mlp = (
        artifacts[
            "grid_mlp"
        ]
    )

    assert (
        grid_mlp
        .named_steps[
            "classifier"
        ]
        .hidden_layer_sizes
        == (
            128,
            64,
            32,
            16,
        )
    )

    assert (
        grid_mlp
        .named_steps[
            "classifier"
        ]
        .activation
        == "tanh"
    )

    # Initial RF.
    grid_rf_initial = (
        artifacts[
            "grid_rf_initial"
        ]
    )

    assert (
        grid_rf_initial.best_params_
        == {
            "classifier__max_depth": 20,
            "classifier__min_samples_split": 2,
            "classifier__n_estimators": 100,
        }
    )

    assert (
        grid_rf_initial.best_score_
        >= 0.32
    )

    # Improved RF.
    grid_rf_improved = (
        artifacts[
            "grid_rf_improved"
        ]
    )

    assert (
        grid_rf_improved.best_params_
        == {
            "classifier__class_weight": None,
            "classifier__max_depth": None,
            "classifier__min_samples_leaf": 1,
            "classifier__min_samples_split": 2,
            "classifier__n_estimators": 300,
        }
    )

    assert (
        grid_rf_improved.best_score_
        >= 0.32
    )


# ============================================================
# CBADVAI: INITIAL TEST-SET EVALUATION
# ============================================================

def _metrics_dict(
    model_names,
    metrics_list
):
    return dict(
        zip(
            model_names,
            metrics_list
        )
    )


def _assert_output_contract(
    results,
    test_size
):
    assert set(
        results.keys()
    ) == {
        "Macro_F1",
        "MAE",
        "QWK",
        "Confusion_Matrix",
    }

    assert np.isfinite(
        results["Macro_F1"]
    )

    assert np.isfinite(
        results["MAE"]
    )

    assert np.isfinite(
        results["QWK"]
    )

    assert (
        0.0
        <= results["Macro_F1"]
        <= 1.0
    )

    assert (
        results["MAE"]
        >= 0.0
    )

    assert (
        -1.0
        <= results["QWK"]
        <= 1.0
    )

    confusion = results[
        "Confusion_Matrix"
    ]

    assert confusion.shape == (
        5,
        5,
    )

    assert (
        confusion >= 0
    ).all()

    assert (
        confusion.sum()
        == test_size
    )


def test_cbadvai_initial_models_test_set(
    system_data,
    model_selection_artifacts
):
    """
    Checkpoint 7:

    Evaluates the selected OLR, MLP, and Initial RF on
    the untouched 20% test set alongside the Dummy baseline.

    This uses the same run_full_evaluation() path as the notebook.
    """

    X_train = system_data[
        "X_train"
    ]

    X_test = system_data[
        "X_test"
    ]

    y_train = system_data[
        "y_train"
    ]

    y_test = system_data[
        "y_test"
    ]

    artifacts = (
        model_selection_artifacts
    )

    model_names, metrics_list = (
        run_full_evaluation(
            X_train,
            y_train,
            X_test,
            y_test,
            artifacts["grid_lr"],
            artifacts["grid_mlp"],
            artifacts["grid_rf_initial"],
            rf_label="Random Forest",
        )
    )

    assert model_names == [
        "Dummy Baseline",
        "Ordinal LR",
        "Optimized MLP",
        "Random Forest",
    ]

    metrics = _metrics_dict(
        model_names,
        metrics_list
    )

    for model_name, gates in (
        INITIAL_GATES.items()
    ):

        _assert_output_contract(
            metrics[model_name],
            len(X_test),
        )

        assert (
            metrics[model_name][
                "Macro_F1"
            ]
            >= gates["min_f1"]
        )

        assert (
            metrics[model_name][
                "MAE"
            ]
            <= gates["max_mae"]
        )

        assert (
            metrics[model_name][
                "QWK"
            ]
            >= gates["min_qwk"]
        )

    # Initial notebook result:
    # RF has the highest Macro F1 among trained models.
    assert (
        metrics["Random Forest"][
            "Macro_F1"
        ]
        >= metrics["Ordinal LR"][
            "Macro_F1"
        ]
    )

    assert (
        metrics["Random Forest"][
            "Macro_F1"
        ]
        >= metrics["Optimized MLP"][
            "Macro_F1"
        ]
    )


# ============================================================
# CBADVAI: FINAL IMPROVED RF
# ============================================================

def test_cbadvai_improved_rf_final_evaluation(
    system_data,
    model_selection_artifacts
):
    """
    Checkpoint 8:

    Evaluates Improved RF using the same final evaluation
    pathway as the notebook, including probability-weighted
    expected-value post-processing.
    """

    X_train = system_data[
        "X_train"
    ]

    X_test = system_data[
        "X_test"
    ]

    y_train = system_data[
        "y_train"
    ]

    y_test = system_data[
        "y_test"
    ]

    artifacts = (
        model_selection_artifacts
    )

    initial_names, initial_metrics_list = (
        run_full_evaluation(
            X_train,
            y_train,
            X_test,
            y_test,
            artifacts["grid_lr"],
            artifacts["grid_mlp"],
            artifacts["grid_rf_initial"],
            rf_label="Random Forest",
        )
    )

    final_names, final_metrics_list = (
        run_full_evaluation(
            X_train,
            y_train,
            X_test,
            y_test,
            artifacts["grid_lr"],
            artifacts["grid_mlp"],
            artifacts["grid_rf_improved"],
            rf_label="Improved RF",
        )
    )

    initial = _metrics_dict(
        initial_names,
        initial_metrics_list,
    )

    final = _metrics_dict(
        final_names,
        final_metrics_list,
    )

    assert final_names == [
        "Dummy Baseline",
        "Ordinal LR",
        "Optimized MLP",
        "Improved RF",
    ]

    improved_rf = final[
        "Improved RF"
    ]

    _assert_output_contract(
        improved_rf,
        len(X_test),
    )

    assert (
        improved_rf["Macro_F1"]
        >= FINAL_RF_GATE["min_f1"]
    )

    assert (
        improved_rf["MAE"]
        <= FINAL_RF_GATE["max_mae"]
    )

    assert (
        improved_rf["QWK"]
        >= FINAL_RF_GATE["min_qwk"]
    )

    # Improved RF dominates LR and MLP.
    for model_name in [
        "Ordinal LR",
        "Optimized MLP",
    ]:

        assert (
            improved_rf["Macro_F1"]
            >= final[model_name][
                "Macro_F1"
            ]
        )

        assert (
            improved_rf["MAE"]
            <= final[model_name][
                "MAE"
            ]
        )

        assert (
            improved_rf["QWK"]
            >= final[model_name][
                "QWK"
            ]
        )

    # Improved RF must improve or maintain all metrics
    # relative to the Initial RF.
    assert (
        improved_rf["Macro_F1"]
        >= initial["Random Forest"][
            "Macro_F1"
        ]
    )

    assert (
        improved_rf["MAE"]
        <= initial["Random Forest"][
            "MAE"
        ]
    )

    assert (
        improved_rf["QWK"]
        >= initial["Random Forest"][
            "QWK"
        ]
    )


# ============================================================
# CBADVAI: FEATURE WEIGHT OUTPUT
# ============================================================

def test_cbadvai_feature_weight_output(
    system_data,
    model_selection_artifacts
):
    """
    Checkpoint 9:

    Verifies final post-training feature weighting across:
        OLR
        Optimized MLP
        Initial RF
        Improved RF
    """

    artifacts = (
        model_selection_artifacts
    )

    weights = (
        compute_summary_feature_weights(
            grid_lr=artifacts[
                "grid_lr"
            ],
            grid_mlp=artifacts[
                "grid_mlp"
            ],
            grid_rf_initial=artifacts[
                "grid_rf_initial"
            ],
            grid_rf_improved=artifacts[
                "grid_rf_improved"
            ],
            X_test=system_data[
                "X_test"
            ],
            y_test=system_data[
                "y_test"
            ],
            feature_names=SELECTED_FEATURES,
        )
    )

    assert isinstance(
        weights,
        pd.DataFrame
    )

    assert weights.shape == (
        11,
        4,
    )

    assert list(
        weights.index
    ) == list(
        SELECTED_FEATURES
    )

    assert list(
        weights.columns
    ) == [
        "Ordinal LR",
        "Optimized MLP",
        "Initial RF",
        "Improved RF",
    ]

    assert not (
        weights
        .isnull()
        .any()
        .any()
    )

    assert np.isfinite(
        weights.to_numpy()
    ).all()

    assert (
        weights >= 0
    ).all().all()

    assert (
        weights <= 1
    ).all().all()

    # Each model's normalized feature weights sum to 1.
    for column in weights.columns:

        assert (
            weights[column].sum()
            == pytest.approx(
                1.0,
                abs=1e-4,
            )
        )

    overall_mean = (
        weights.mean(axis=1)
    )

    assert (
        overall_mean.sum()
        == pytest.approx(
            1.0,
            abs=1e-4,
        )
    )

    # Notebook-established top overall drivers.
    top_three = set(
        overall_mean.nlargest(3).index
    )

    assert top_three == {
        "Time_Friends",
        "Quality_Lecturer",
        "Time_SocicalMedia",
    }