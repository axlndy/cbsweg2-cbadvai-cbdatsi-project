# tests/test_eda_plots.py
import pytest
import matplotlib.pyplot as plt
from unittest.mock import patch

from src.cbdatsi.pipeline import perform_feature_engineering
from src.cbdatsi.eda_plots import (
    plot_demographics,
    plot_behavioral_boxplots,
    plot_socioeconomic_conditional,
    plot_institutional_heatmap,
    plot_mindset_heatmaps
)


@pytest.fixture
def engineered_data(dummy_student_data):
    """Provides feature-engineered data required by the EDA plotting functions."""
    return perform_feature_engineering(dummy_student_data)


@patch("matplotlib.pyplot.show")
def test_plot_demographics_structure(mock_show, engineered_data):
    """
    Verifies that the demographics plot creates three axes
    with the expected titles.
    """
    plot_demographics(engineered_data)

    fig = plt.gcf()

    assert len(fig.axes) == 3
    assert fig.axes[0].get_title() == "Distribution by Year Level"
    assert fig.axes[1].get_title() == "Distribution by Gender"
    assert fig.axes[2].get_title() == "Baseline GPA Distribution"

    mock_show.assert_called_once()
    plt.close("all")


@patch("matplotlib.pyplot.show")
def test_plot_behavioral_boxplots_structure(mock_show, engineered_data):
    """
    Verifies that the behavioral boxplot function creates two axes
    with the expected titles and y-axis labels.
    """
    plot_behavioral_boxplots(engineered_data)

    fig = plt.gcf()

    assert len(fig.axes) == 2

    assert (
        fig.axes[0].get_title()
        == "Study Time Distribution Across GPA Brackets"
    )
    assert (
        fig.axes[1].get_title()
        == "Social Media Time Distribution Across GPA Brackets"
    )

    assert fig.axes[0].get_ylabel() == "Study Time Level (1-5 Scale)"
    assert fig.axes[1].get_ylabel() == "Social Media Level (1-5 Scale)"

    mock_show.assert_called_once()
    plt.close("all")


@patch("matplotlib.pyplot.show")
def test_plot_socioeconomic_conditional_structure(
    mock_show, engineered_data
):
    """
    Verifies that the socioeconomic conditional plots create two axes
    with the expected titles and percentage-based y-axis labels.
    """
    plot_socioeconomic_conditional(engineered_data)

    fig = plt.gcf()

    assert len(fig.axes) == 2

    assert (
        fig.axes[0].get_title()
        == "Household Wealth Breakdown Per GPA Level"
    )
    assert (
        fig.axes[1].get_title()
        == "Policy Support Breakdown Per GPA Level"
    )

    assert fig.axes[0].get_ylabel() == "Percentage within GPA Tier (%)"
    assert fig.axes[1].get_ylabel() == "Percentage within GPA Tier (%)"

    mock_show.assert_called_once()
    plt.close("all")


@patch("matplotlib.pyplot.show")
def test_plot_institutional_heatmap_structure(
    mock_show, engineered_data
):
    """
    Verifies that the institutional analysis creates a heatmap
    with the expected title.
    """
    plot_institutional_heatmap(engineered_data)

    fig = plt.gcf()

    assert len(fig.axes) >= 1
    assert fig.axes[0].get_title() == "Institutional Factors vs. Student GPA"

    mock_show.assert_called_once()
    plt.close("all")


@patch("matplotlib.pyplot.show")
def test_plot_mindset_heatmaps_structure(
    mock_show, engineered_data
):
    """
    Verifies that the mindset analysis creates two heatmaps
    with the expected titles.
    """
    plot_mindset_heatmaps(engineered_data)

    fig = plt.gcf()

    assert len(fig.axes) >= 2

    assert (
        fig.axes[0].get_title()
        == "GPA Distribution by Study Methods (Internal)"
    )
    assert (
        fig.axes[1].get_title()
        == "GPA Distribution by Class Competition (External)"
    )

    mock_show.assert_called_once()
    plt.close("all")