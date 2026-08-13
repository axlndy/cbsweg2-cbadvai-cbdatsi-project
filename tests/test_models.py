import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from imblearn.pipeline import Pipeline as ImbPipeline

# Adjust imports based on your exact file structure
from src.cbadvai.models import (
    build_ordinal_lr_pipeline, 
    build_mlp_pipeline, 
    build_rf_pipeline,
    tune_initial_rf
)

@pytest.fixture
def micro_dataset():
    """Generates a tiny, 20-row dataset strictly for testing model execution without hanging the CI pipeline."""
    X, y = make_classification(
        n_samples=20, 
        n_features=11, 
        n_informative=5, 
        n_classes=3, 
        random_state=42
    )
    # y must start at 1 to mimic your 1-5 GPA scale
    return pd.DataFrame(X), pd.Series(y + 1)

def test_pipeline_builders():
    """Validates that the pipeline builders correctly assemble SMOTE and the designated classifier."""
    lr_pipe = build_ordinal_lr_pipeline()
    mlp_pipe = build_mlp_pipeline()
    rf_pipe = build_rf_pipeline()
    
    # 1. Verify they return Imblearn Pipelines (so SMOTE works correctly)
    assert isinstance(lr_pipe, ImbPipeline)
    assert isinstance(mlp_pipe, ImbPipeline)
    assert isinstance(rf_pipe, ImbPipeline)
    
    # 2. Verify the steps are named correctly as expected by GridSearchCV
    assert 'smote' in lr_pipe.named_steps
    assert 'classifier' in lr_pipe.named_steps

def test_tune_initial_rf_execution(micro_dataset):
    """
    Validates that the Random Forest tuning function executes without syntax or dimensional errors.
    Uses cv=2 and the micro_dataset to ensure it runs in milliseconds.
    """
    X_micro, y_micro = micro_dataset
    
    # We pass a minimal cv=2 so it doesn't do a massive grid search
    grid_rf = tune_initial_rf(X_micro, y_micro, cv=2)
    
    # Verify the object returned is fitted and contains cross-validation results
    assert hasattr(grid_rf, 'cv_results_'), "GridSearchCV did not return a fitted model."
    assert hasattr(grid_rf, 'best_estimator_'), "GridSearchCV failed to find a best estimator."
    
    # Verify the model can successfully execute a prediction
    preds = grid_rf.predict(X_micro)
    assert len(preds) == len(X_micro), "Model prediction output length mismatch."