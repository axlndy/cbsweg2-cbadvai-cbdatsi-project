"""
Core modeling, metrics, and preprocessing package for CBADVAI ordinal classification pipeline.
"""
from .preprocessing import load_and_preprocess_data, get_train_test_split
from .metrics import evaluate_ordinal_model, evaluate_dummy_baseline
from .models import (
    FrankHallOrdinalClassifier, 
    build_ordinal_lr_pipeline, 
    build_mlp_pipeline, 
    build_rf_pipeline
)