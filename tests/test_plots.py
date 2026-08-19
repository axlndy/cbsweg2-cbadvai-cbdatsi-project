import pytest
import matplotlib
matplotlib.use('Agg')  # Headless backend for CI/CD environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cbadvai.plots import (
    plot_model_metrics_comparison,
    plot_confusion_matrices,
    plot_feature_weights_heatmap,
    plot_average_feature_weights
)

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

def test_plot_model_metrics_comparison_execution():
    """Smoke test: Verifies model metric bar chart comparison renders without crashing."""
    metrics_list = [
        {'Macro_F1': 0.82, 'MAE': 0.15, 'QWK': 0.78},
        {'Macro_F1': 0.85, 'MAE': 0.12, 'QWK': 0.81}
    ]
    model_names = ['Ordinal LR', 'Optimized MLP']

    # Test both 2-metric and 3-metric (include_qwk) modes
    plot_model_metrics_comparison(metrics_list, model_names, include_qwk=False)
    plot_model_metrics_comparison(metrics_list, model_names, include_qwk=True)


def test_plot_confusion_matrices_execution():
    """Smoke test: Verifies confusion matrix grid plotting executes across models."""
    metrics_list = [
        {'Confusion_Matrix': np.eye(5, dtype=int)},
        {'Confusion_Matrix': np.eye(5, dtype=int)}
    ]
    model_names = ['Model A', 'Model B']

    plot_confusion_matrices(metrics_list, model_names)


def test_plot_feature_weights_plots_execution():
    """Smoke test: Verifies feature importance heatmap and average bar chart render without error."""
    mock_weights = pd.DataFrame(
        np.random.rand(11, 4), 
        index=EXPECTED_FEATURES, 
        columns=['Ordinal LR', 'Optimized MLP', 'Initial RF', 'Improved RF']
    )

    plot_feature_weights_heatmap(mock_weights)
    plot_average_feature_weights(mock_weights)
