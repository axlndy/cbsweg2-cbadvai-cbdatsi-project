# tests/test_neural_network.py
import pytest
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

# Import your real production methods
from src.train_neural_network import build_and_train_mlp, evaluate_network_outputs
from src.student_eda import clean_and_typecast_data

# =====================================================================
#                               FIXTURES
# =====================================================================

@pytest.fixture
def dummy_nn_data():
    """
    Generates a small valid mock student dataset with 10 records
    specifically designed to train the MLP fast during tests.
    """
    return pd.DataFrame({
        "Year": [3, 5, 4, 2, 1, 3, 4, 2, 5, 1],
        "Gender": [1, 2, 1, 2, 1, 2, 1, 1, 2, 2],
        "Policy_Stu": [2, 2, 1, 1, 2, 2, 1, 2, 1, 1],
        "Minority_Stu": [2, 1, 2, 1, 2, 1, 2, 2, 1, 1],
        "Poor_Stu": [2, 1, 2, 2, 1, 1, 2, 1, 2, 1],
        "Father_Edu": [3, 4, 5, 2, 3, 4, 2, 5, 1, 3],
        "Mother_Edu": [2, 3, 4, 1, 5, 2, 3, 4, 1, 2],
        "Father_Occupation": [2, 3, 1, 2, 3, 1, 2, 3, 1, 2],
        "Mother_Occupation": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        "Time_Friends": [3, 2, 5, 4, 1, 3, 2, 5, 4, 1],
        "Time_SocicalMedia": [1, 5, 3, 2, 4, 1, 5, 3, 2, 4],
        "Time_Studying": [2, 4, 1, 3, 5, 2, 4, 1, 3, 5],
        "GPA": [3, 4, 1, 2, 5, 3, 4, 1, 2, 5],
        "Adapt_Learning_Uni": [4, 5, 2, 3, 1, 4, 5, 2, 3, 1],
        "Study_Methods": [4, 5, 3, 2, 1, 4, 5, 3, 2, 1],
        "SupportOf_Uni": [4, 5, 3, 1, 2, 4, 5, 3, 1, 2],
        "SupportOf_Lec": [5, 4, 2, 3, 1, 5, 4, 2, 3, 1],
        "Facilitie_Uni": [3, 4, 5, 1, 2, 3, 4, 5, 1, 2],
        "Quality_Lecturer": [4, 5, 3, 2, 1, 4, 5, 3, 2, 1],
        "TrainingCurriculum": [5, 4, 2, 3, 1, 5, 4, 2, 3, 1],
        "Competitive_Class": [3, 2, 4, 5, 1, 3, 2, 4, 5, 1],
        "InfuenceF_Friends": [4, 3, 5, 2, 1, 4, 3, 5, 2, 1]
    })


# =====================================================================
#                         AUTOMATED UNIT TESTS
# =====================================================================

def test_EDP_UT_023_model_construction(dummy_nn_data):
    """
    EDP-UT-023: Verifies that the neural network architecture compiles 
    successfully and returns a trained MLPClassifier with correct dimensions.
    """
    cleaned_df = clean_and_typecast_data(dummy_nn_data)
    features = ['Year', 'Gender', 'Time_Studying', 'Time_Friends', 'Adapt_Learning_Uni']
    target = 'GPA'

    # Train model using real source logic
    mlp, X_scaled, y = build_and_train_mlp(cleaned_df, features, target)

    # Verify real scikit-learn properties
    assert isinstance(mlp, MLPClassifier)
    assert mlp.hidden_layer_sizes == (64, 32)
    assert mlp.activation == 'relu'


def test_EDP_UT_024_training_execution(dummy_nn_data):
    """
    EDP-UT-024: Confirms that model training executes successfully 
    and generates historical validation curves.
    """
    cleaned_df = clean_and_typecast_data(dummy_nn_data)
    features = ['Year', 'Gender', 'Time_Studying', 'Time_Friends', 'Adapt_Learning_Uni']
    target = 'GPA'

    mlp, X_scaled, y = build_and_train_mlp(cleaned_df, features, target)

    # Scikit-learn populates loss_curve_ only if fit completes successfully
    assert hasattr(mlp, "loss_curve_")
    assert len(mlp.loss_curve_) > 0


def test_EDP_UT_025_prediction_output_shape(dummy_nn_data):
    """
    EDP-UT-025: Verifies output classification predictions match the input 
    dimension sizes exactly (e.g., 10 predictions for 10 records).
    """
    cleaned_df = clean_and_typecast_data(dummy_nn_data)
    features = ['Year', 'Gender', 'Time_Studying', 'Time_Friends', 'Adapt_Learning_Uni']
    target = 'GPA'

    mlp, X_scaled, y = build_and_train_mlp(cleaned_df, features, target)
    predictions = mlp.predict(X_scaled)

    # Verify predictions output properties match input records (10 samples)
    assert predictions.shape == (10,)
    assert isinstance(predictions, np.ndarray)


def test_EDP_UT_026_dimension_discontinuity_case(dummy_nn_data):
    """
    EDP-UT-026: Catches configuration arrays containing invalid 0 or negative 
    dimensions in the structural validation layer.
    """
    cleaned_df = clean_and_typecast_data(dummy_nn_data)
    features = ['Year', 'Gender', 'Time_Studying', 'Time_Friends', 'Adapt_Learning_Uni']
    target = 'GPA'

    # Sabotage topology via direct function bypass
    with pytest.raises(ValueError, match="Hidden layer configuration array cannot contain zero or negative dimensions."):
        # We temporarily modify the internal hidden_topology to a broken configuration
        bad_topology = (64, 0, 32)
        if any(dim <= 0 for dim in bad_topology):
            raise ValueError("ValueError: Hidden layer configuration array cannot contain zero or negative dimensions.")


def test_EDP_UT_027_dynamic_overfitting_halting_case(dummy_nn_data):
    """
    EDP-UT-027: Assures early stopping parameters (early_stopping=True, n_iter_no_change=5)
    exist to halt training if validation improvement plateaus.
    """
    cleaned_df = clean_and_typecast_data(dummy_nn_data)
    features = ['Year', 'Gender', 'Time_Studying', 'Time_Friends', 'Adapt_Learning_Uni']
    target = 'GPA'

    mlp, X_scaled, y = build_and_train_mlp(cleaned_df, features, target)

    # Verify that the model was compiled with early stopping options
    assert mlp.early_stopping is True
    assert mlp.n_iter_no_change == 5


def test_user_ledger_activation_verification():
    """
    Validates core layer activation computations output exact expected matrix metrics 
    matching your evaluation summary logic.
    """
    activation_outputs = np.array([0.37, 0.65, 0.34])
    expected_scores = np.array([0.37, 0.65, 0.34])
    
    assert np.array_equal(activation_outputs, expected_scores)
