# tests/test_regression.py
import os

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


# ============================================================
# FIXTURE
# ============================================================

@pytest.fixture(scope="module")
def actual_dataset(tmp_path_factory):
    """
    Loads the actual CBDATSI dataset used by the notebook.

    A temporary cache is used so that the regression tests
    always begin from the repository's actual Excel dataset
    rather than potentially loading a stale cache file.
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
        "dataset_cache.pkl"
    )

    # --------------------------------------------------------
    # Reproduce the notebook preprocessing pipeline
    # --------------------------------------------------------

    df = load_and_cache_dataset(
        ACTUAL_DATA_PATH,
        cache_path
    )

    validate_dataset(df)

    df = clean_and_typecast_data(df)

    df = perform_feature_engineering(df)

    return df


# ============================================================
# DATASET REGRESSION
# ============================================================

def test_regression_cbdatsi_data_integrity(
    actual_dataset
):
    """
    Verifies that the actual CBDATSI dataset still matches
    the documented notebook baseline.

    Expected:
        observations = 2170
        missing values = 0
        duplicate rows = 226
    """

    assert len(actual_dataset) == 2170, (
        "Dataset row count drifted from 2170."
    )

    assert (
        actual_dataset.isnull().sum().sum()
        == 0
    ), (
        "Null values detected in the actual CBDATSI dataset."
    )

    assert (
        actual_dataset.duplicated().sum()
        == 226
    ), (
        "Duplicate row count drifted from 226."
    )


def test_regression_cbdatsi_clustering_specification(
    actual_dataset
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


# ============================================================
# K-MEANS REGRESSION
# ============================================================

def test_regression_cbdatsi_kmeans_silhouette(
    actual_dataset
):
    """
    Verifies that the K-Means analysis reproduces the
    notebook's silhouette score.

    Conditions:
        n = 2170
        clustering variables = 11
        k = 3

    Expected:
        silhouette ≈ 0.1456
    """

    features = get_clustering_features()

    assert len(features) == 11

    clustered_df, cluster_summary = (
        run_kmeans_clustering(
            actual_dataset,
            n_clusters=3
        )
    )

    # Verify the clustering did not change the number
    # of observations.
    assert len(clustered_df) == 2170

    # Verify k=3 actually produced three clusters.
    assert (
        clustered_df["Cluster"].nunique()
        == 3
    )

    # Verify the expected cluster summary structure.
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
        abs=1e-4
    ), (
        "Silhouette score drifted from the "
        "CBDATSI notebook baseline."
    )


# ============================================================
# CHI-SQUARE REGRESSION
# ============================================================

def test_regression_cbdatsi_chisquare_results(
    actual_dataset
):
    """
    Verifies that the complete CBDATSI clustering-to-inference
    pipeline reproduces the notebook's Chi-Square results.

    Conditions:
        n = 2170
        clustering variables = 11
        k = 3
        inference = Cluster × GPA

    Expected:
        Chi-Square ≈ 43.2357
        df = 8
        p ≈ 7.9306e-7
        minimum expected frequency ≈ 19.78
    """

    features = get_clustering_features()

    assert len(features) == 11

    clustered_df, _ = (
        run_kmeans_clustering(
            actual_dataset,
            n_clusters=3
        )
    )

    chi2, p_val, dof, table = (
        perform_chisquare_independence(
            clustered_df,
            target_col="GPA",
            cluster_col="Cluster",
        )
    )

    # --------------------------------------------------------
    # 1. Statistical outputs
    # --------------------------------------------------------

    assert chi2 == pytest.approx(
        43.2357,
        abs=1e-4
    ), (
        "Chi-Square statistic drifted from "
        "the notebook baseline."
    )

    assert dof == 8, (
        "Degrees of freedom drifted from "
        "the notebook baseline."
    )

    assert p_val == pytest.approx(
        7.9306e-7,
        rel=1e-4
    ), (
        "p-value drifted from the notebook baseline."
    )

    # --------------------------------------------------------
    # 2. Contingency-table dimensions
    # --------------------------------------------------------

    assert table.shape == (3, 5)

    # --------------------------------------------------------
    # 3. Expected-frequency assumption
    # --------------------------------------------------------

    _, _, _, expected_freq = (
        stats.chi2_contingency(table)
    )

    assert expected_freq.min() == pytest.approx(
        19.78,
        abs=1e-2
    )

    assert expected_freq.min() >= 5

    # --------------------------------------------------------
    # 4. Exact observed contingency table
    # --------------------------------------------------------

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
            name="Cluster"
        )
    )

    expected_table.columns.name = "GPA"

    pd.testing.assert_frame_equal(
        table,
        expected_table,
        check_index_type=False
    )