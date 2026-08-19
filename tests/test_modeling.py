# tests/test_modeling.py
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from unittest.mock import patch

from src.cbdatsi.modeling import (
    get_clustering_features,
    preprocess_clustering_data,
    run_kmeans_clustering,
    evaluate_clusters,
    plot_elbow_curve,
    plot_cluster_pca,
    plot_cluster_profiles,
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
    X = preprocess_clustering_data(dummy_student_data)

    assert X.shape == (
        len(dummy_student_data),
        11
    )

    assert list(X.columns) == EXPECTED_CLUSTERING_FEATURES


def test_policy_student_encoding(
    dummy_student_data
):
    """
    Verifies that Policy_Stu is converted from:
        1 -> 1
        2 -> 0
    """
    X = preprocess_clustering_data(dummy_student_data)

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

    preprocess_clustering_data(dummy_student_data)

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
    clustered_df, summary = run_kmeans_clustering(
        dummy_student_data,
        n_clusters=3
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
    clustered_df, summary = run_kmeans_clustering(
        dummy_student_data,
        n_clusters=2
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


def test_run_kmeans_does_not_modify_original(
    dummy_student_data
):
    """
    Verifies that K-Means clustering does not modify the
    original input DataFrame.
    """
    original = dummy_student_data.copy()

    run_kmeans_clustering(
        dummy_student_data,
        n_clusters=3
    )

    pd.testing.assert_frame_equal(
        dummy_student_data,
        original
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

    assert set(summary.columns) == expected_columns

    assert len(summary) == 3


def test_cluster_summary_has_numeric_values(
    dummy_student_data
):
    """
    Verifies that all cluster summary values are numeric
    because the summary represents feature means.
    """
    _, summary = run_kmeans_clustering(
        dummy_student_data,
        n_clusters=3
    )

    assert all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in summary.dtypes
    )

    assert summary.notna().all().all()


def test_cluster_labels_are_valid(
    dummy_student_data
):
    """
    Verifies that generated cluster labels are valid integer
    labels ranging from 0 to k-1.
    """
    clustered_df, _ = run_kmeans_clustering(
        dummy_student_data,
        n_clusters=3
    )

    labels = clustered_df["Cluster"]

    assert pd.api.types.is_integer_dtype(labels)

    assert set(labels.unique()).issubset(
        {0, 1, 2}
    )


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
    clustered_df, _ = run_kmeans_clustering(
        dummy_student_data,
        n_clusters=3
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
    clustered_df, _ = run_kmeans_clustering(
        dummy_student_data,
        n_clusters=3
    )

    score = evaluate_clusters(
        clustered_df
    )

    assert -1.0 <= score <= 1.0


# ============================================================
# CLUSTERING PLOTS
# ============================================================

@patch("matplotlib.pyplot.show")
def test_plot_elbow_curve_structure(
    mock_show,
    dummy_student_data
):
    """
    Verifies that the Elbow Method creates a plot with the
    expected title, axes, and labels.
    """
    plot_elbow_curve(
        dummy_student_data,
        max_k=4
    )

    fig = plt.gcf()

    assert len(fig.axes) == 1

    ax = fig.axes[0]

    assert ax.get_title() == "Elbow Method For Optimal k"
    assert ax.get_xlabel() == "Number of Clusters (k)"
    assert ax.get_ylabel() == (
        "Within-Cluster Sum of Squares (WCSS)"
    )

    # The plot should contain one WCSS value for each k.
    assert len(ax.lines) == 1
    assert len(ax.lines[0].get_xdata()) == 4
    assert len(ax.lines[0].get_ydata()) == 4

    mock_show.assert_called_once()

    plt.close("all")


@patch("matplotlib.pyplot.show")
def test_plot_elbow_curve_wcss_values_are_valid(
    mock_show,
    dummy_student_data
):
    """
    Verifies that the Elbow Method produces finite,
    non-negative WCSS values.
    """
    plot_elbow_curve(
        dummy_student_data,
        max_k=4
    )

    fig = plt.gcf()
    ax = fig.axes[0]

    wcss_values = ax.lines[0].get_ydata()

    assert len(wcss_values) == 4
    assert all(value >= 0 for value in wcss_values)
    assert all(
        pd.notna(value) for value in wcss_values
    )

    mock_show.assert_called_once()

    plt.close("all")


@patch("matplotlib.pyplot.show")
def test_plot_cluster_pca_structure(
    mock_show,
    dummy_student_data
):
    """
    Verifies that the PCA clustering visualization creates
    a 2D plot with the expected title and axis labels.
    """
    clustered_df, _ = run_kmeans_clustering(
        dummy_student_data,
        n_clusters=2
    )

    plot_cluster_pca(
        clustered_df
    )

    fig = plt.gcf()

    assert len(fig.axes) == 1

    ax = fig.axes[0]

    assert ax.get_title() == (
        "2D PCA Visualization of Student Clusters"
    )

    assert "Principal Component 1" in (
        ax.get_xlabel()
    )

    assert "Principal Component 2" in (
        ax.get_ylabel()
    )

    mock_show.assert_called_once()

    plt.close("all")


@patch("matplotlib.pyplot.show")
def test_plot_cluster_pca_contains_cluster_points(
    mock_show,
    dummy_student_data
):
    """
    Verifies that the PCA visualization contains plotted
    observations corresponding to the input students.
    """
    clustered_df, _ = run_kmeans_clustering(
        dummy_student_data,
        n_clusters=2
    )

    plot_cluster_pca(
        clustered_df
    )

    fig = plt.gcf()
    ax = fig.axes[0]

    # seaborn creates one or more PathCollections for
    # the plotted cluster observations.
    collections = ax.collections

    assert len(collections) >= 1

    total_points = sum(
        len(collection.get_offsets())
        for collection in collections
    )

    assert total_points == len(
        clustered_df
    )

    mock_show.assert_called_once()

    plt.close("all")


@patch("matplotlib.pyplot.show")
def test_plot_cluster_profiles_structure(
    mock_show,
    dummy_student_data
):
    """
    Verifies that the cluster profile plot creates a grouped
    bar chart with the expected title and y-axis label.
    """
    _, cluster_summary = run_kmeans_clustering(
        dummy_student_data,
        n_clusters=3
    )

    plot_cluster_profiles(
        cluster_summary
    )

    fig = plt.gcf()

    assert len(fig.axes) == 1

    ax = fig.axes[0]

    assert ax.get_title() == (
        "Feature Means and GPA by Student Cluster"
    )

    assert ax.get_ylabel() == "Mean Value"

    mock_show.assert_called_once()

    plt.close("all")


@patch("matplotlib.pyplot.show")
def test_plot_cluster_profiles_contains_all_features(
    mock_show,
    dummy_student_data
):
    """
    Verifies that the cluster profile chart contains one
    category for every feature and GPA in the cluster summary.
    """
    _, cluster_summary = run_kmeans_clustering(
        dummy_student_data,
        n_clusters=3
    )

    plot_cluster_profiles(
        cluster_summary
    )

    fig = plt.gcf()
    ax = fig.axes[0]

    expected_feature_count = len(
        cluster_summary.columns
    )

    # Each feature/GPA is represented by one x-axis category.
    assert len(ax.get_xticks()) == expected_feature_count

    mock_show.assert_called_once()

    plt.close("all")