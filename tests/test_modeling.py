import pandas as pd
import pytest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.cbdatsi.modeling import (
    get_clustering_features,
    preprocess_clustering_data,
    run_kmeans_clustering,
    evaluate_clusters,
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
# FEATURE SELECTION
# ============================================================

def test_get_clustering_features():
    """
    Verifies that exactly the 11 variables used by the CBDATSI
    K-Means analysis are selected.
    """

    features = get_clustering_features()

    assert len(features) == 11
    assert features == EXPECTED_CLUSTERING_FEATURES
    assert len(features) == len(set(features))


# ============================================================
# CLUSTERING PREPROCESSING
# ============================================================

def test_preprocess_clustering_data_uses_11_features(
    dummy_student_data
):
    """
    Verifies that clustering preprocessing produces exactly
    the 11-feature model input.
    """

    X = preprocess_clustering_data(
        dummy_student_data
    )

    assert X.shape == (
        len(dummy_student_data),
        11
    )

    assert list(X.columns) == (
        EXPECTED_CLUSTERING_FEATURES
    )


def test_policy_student_encoding(
    dummy_student_data
):
    """
    Verifies that Policy_Stu is converted from:
        1 -> 1
        2 -> 0
    """

    X = preprocess_clustering_data(
        dummy_student_data
    )

    assert X["Policy_Stu"].tolist() == [
        0, 0, 1, 1, 0
    ]


def test_preprocess_does_not_modify_original(
    dummy_student_data
):
    """
    Verifies that clustering preprocessing does not alter
    the original DataFrame.
    """

    original = dummy_student_data.copy()

    preprocess_clustering_data(
        dummy_student_data
    )

    pd.testing.assert_frame_equal(
        dummy_student_data,
        original
    )


# ============================================================
# K-MEANS
# ============================================================

def test_run_kmeans_three_clusters(
    dummy_student_data
):
    """
    Verifies the normal CBDATSI K-Means configuration:
    11 features and k=3.
    """

    clustered_df, summary = (
        run_kmeans_clustering(
            dummy_student_data,
            n_clusters=3
        )
    )

    assert "Cluster" in clustered_df.columns

    assert len(clustered_df) == len(
        dummy_student_data
    )

    assert clustered_df["Cluster"].nunique() == 3

    assert len(summary) == 3


def test_run_kmeans_two_clusters(
    dummy_student_data
):
    """
    Verifies that K-Means can also operate with a valid
    alternative cluster count.
    """

    clustered_df, summary = (
        run_kmeans_clustering(
            dummy_student_data,
            n_clusters=2
        )
    )

    assert clustered_df["Cluster"].nunique() == 2
    assert len(summary) == 2


@pytest.mark.parametrize(
    "invalid_k",
    [0, -1, 6, 10]
)
def test_run_kmeans_invalid_cluster_count(
    dummy_student_data,
    invalid_k
):
    """
    Verifies that invalid cluster counts are rejected.
    """

    with pytest.raises(ValueError):
        run_kmeans_clustering(
            dummy_student_data,
            n_clusters=invalid_k
        )


# ============================================================
# CLUSTER SUMMARY
# ============================================================

def test_cluster_summary_contains_expected_features(
    dummy_student_data
):
    """
    Verifies that the cluster summary contains the 11
    clustering variables plus GPA.
    """

    _, summary = run_kmeans_clustering(
        dummy_student_data,
        n_clusters=3
    )

    expected_columns = set(
        EXPECTED_CLUSTERING_FEATURES
    ) | {"GPA"}

    assert set(summary.columns) == (
        expected_columns
    )

    assert len(summary) == 3


# ============================================================
# SILHOUETTE
# ============================================================

def test_evaluate_clusters_matches_sklearn(
    dummy_student_data
):
    """
    Verifies that evaluate_clusters() produces the same
    silhouette score as an independent sklearn calculation.
    """

    clustered_df, _ = (
        run_kmeans_clustering(
            dummy_student_data,
            n_clusters=3
        )
    )

    X = preprocess_clustering_data(
        clustered_df
    )

    X_scaled = StandardScaler().fit_transform(X)

    expected_score = silhouette_score(
        X_scaled,
        clustered_df["Cluster"]
    )

    actual_score = evaluate_clusters(
        clustered_df
    )

    assert actual_score == pytest.approx(
        expected_score
    )


def test_evaluate_clusters_range(
    dummy_student_data
):
    """
    Verifies that the silhouette score falls within its
    mathematical range of -1 to 1.
    """

    clustered_df, _ = (
        run_kmeans_clustering(
            dummy_student_data,
            n_clusters=3
        )
    )

    score = evaluate_clusters(
        clustered_df
    )

    assert -1.0 <= score <= 1.0