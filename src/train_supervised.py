# src/train_supervised.py
import os
import pandas as pd
import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel
from student_eda import load_and_cache_dataset, clean_and_typecast_data

def train_ordinal_model(df: pd.DataFrame, feature_cols: list, target_col: str):
    """Trains an Ordinal Logistic Regression model using statsmodels."""
    print(f"\n--- Initializing Ordinal Logistic Regression Training ---")
    X = df[feature_cols]
    y = df[target_col]
    
    # Check for multicollinearity (EDP-UT-022 scenario)
    # statsmodels will throw a Singular Matrix error or fail to converge if features are perfectly correlated
    
    # Initialize the Ordinal Model (Logit link function matches Ordinal Logistic Regression)
    model = OrderedModel(y, X, distr='logit')
    
    print("Optimizing parameter weights (Fitting model)...")
    # fit() terminates optimization and returns the trained results object
    res = model.fit(method='bfgs', disp=False) 
    
    return res

def evaluate_model_predictions(model_res, df: pd.DataFrame, feature_cols: list) -> None:
    """Generates predictions and probability distributions from the trained model."""
    print("\n--- Evaluating Model Predictions & Probabilities ---")
    X_test = df[feature_cols]
    
    # 1. Predict Probabilities (EDP-UT-021)
    # Returns an array of probabilities for each target class (Poor, Average, Fair, Good, Excellent)
    prob_distributions = model_res.model.predict(model_res.params, exog=X_test)
    print(f"Probability distribution matrix shape: {prob_distributions.shape}")
    print("Sample probability distribution for the first 3 students (rows sum to 1.0):")
    print(pd.DataFrame(prob_distributions[:3]).round(4).to_string())
    
    # 2. Predict Discrete Classes (EDP-UT-020)
    # Pick the class index with the highest probability
    predicted_classes = np.argmax(prob_distributions, axis=1) + 1 # +1 to shift index 0-4 to class 1-5
    print(f"\nGenerated discrete class prediction array shape: {predicted_classes.shape}")
    print(f"Sample class predictions for the first 10 students: {predicted_classes[:10]}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    RAW_DATA = os.path.join(DATA_DIR, "Database paper.xlsx")
    CACHE_DATA = os.path.join(DATA_DIR, "dataset_cache.pkl")

    # Define features based on your dataset structure
    # Tracking how academic and behavioral variables influence the target variable (GPA)
    FEATURES = ['Year', 'Gender', 'Time_Studying', 'Time_Friends', 'Adapt_Learning_Uni']
    TARGET = 'GPA'

    try:
        # Load and clean via your existing modular pipeline
        raw_df = load_and_cache_dataset(RAW_DATA, CACHE_DATA)
        cleaned_df = clean_and_typecast_data(raw_df)
        
        # Train Model (Triggers EDP-UT-018 & EDP-UT-019)
        model_results = train_ordinal_model(cleaned_df, FEATURES, TARGET)
        
        # Display Convergence Summary details
        print("\n==============================================")
        print("MODULE 6: MODEL TRAINING CONVERGENCE SUMMARY")
        print("==============================================")
        print(f"Optimization Completed Successfully: {model_results.mle_retvals.get('converged', 'N/A')}")
        print(f"Log-Likelihood Function Value: {model_results.llf:.4f}")
        print(f"Number of Iterations Required: {model_results.mle_retvals.get('iterations')}")
        # Evaluate model outcomes (Triggers EDP-UT-020 & EDP-UT-021)
        evaluate_model_predictions(model_results, cleaned_df, FEATURES)

    except Exception as e:
        print(f"Execution Error during Supervised Training: {e}")