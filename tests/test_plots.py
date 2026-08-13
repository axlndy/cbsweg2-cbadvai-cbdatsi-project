import pytest
import matplotlib
matplotlib.use('Agg')  # Headless backend for CI/CD environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cbadvai.plots import (
    plot_gpa_distribution,
    plot_feature_weights_comparison,
    plot_confusion_matrices
)
from src.cbadvai.preprocessing import SELECTED_FEATURES

# Canonical expected feature contract
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
# FIXTURES & TEARDOWN
# ==============================================================================

@pytest.fixture(autouse=True)
def cleanup_matplotlib_figures():
    """Autouse fixture: Automatically closes all figures after every test to prevent memory leaks."""
    yield
    plt.close('all')


# ==============================================================================
# VISUALIZATION TESTS
# ==============================================================================

def test_plot_gpa_distribution_execution(synthetic_survey_df):
    """Smoke test: Verifies GPA distribution plot executes and returns a valid Figure."""
    y = synthetic_survey_df['GPA']
    fig = plot_gpa_distribution(y)

    assert fig is not None
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1


def test_plot_confusion_matrices_execution():
    """Smoke test: Verifies confusion matrix plotting executes across 4 grid subplots."""
    cm_dummy = np.eye(5, dtype=int)
    cm_lr = np.eye(5, dtype=int)
    cm_mlp = np.eye(5, dtype=int)
    cm_rf = np.eye(5, dtype=int)

    fig = plot_confusion_matrices(cm_dummy, cm_lr, cm_mlp, cm_rf)

    assert fig is not None
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 4

    # Verify that titles are populated on subplots
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert len(titles) >= 4, "Subplots are missing title labels!"


def test_plot_feature_weights_comparison_execution():
    """Smoke test: Verifies feature weight comparison plot renders all 11 features without error."""
    mock_weights = pd.DataFrame(
        np.random.rand(11, 4), 
        index=EXPECTED_FEATURES, 
        columns=['Ordinal LR', 'MLP', 'RF (Initial)', 'RF (Tuned)']
    )

    fig = plot_feature_weights_comparison(mock_weights)

    assert fig is not None
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
