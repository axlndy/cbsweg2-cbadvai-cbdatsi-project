import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, cohen_kappa_score, confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from src.cbadvai.models import predict_ordinal_expected_value


def evaluate_ordinal_model(y_true, y_pred, model_name="Model"):
    """
    Computes performance metrics specifically tailored for ordinal classification:
    Macro F1, MAE, and Quadratic Weighted Kappa (QWK).
    """
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    mae = mean_absolute_error(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    print(f"--- Evaluation Metrics: {model_name} ---")
    print(f"Macro-Averaged F1-Score: {macro_f1:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Quadratic Weighted Kappa (QWK): {qwk:.4f}\n")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("-" * 40)
    
    return {
        "Macro_F1": macro_f1,
        "MAE": mae,
        "QWK": qwk,
        "Confusion_Matrix": conf_matrix
    }


def evaluate_dummy_baseline(X_train, y_train, X_test, y_test, strategy="most_frequent"):
    dummy = DummyClassifier(strategy=strategy, random_state=42)
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)
    return evaluate_ordinal_model(y_test, y_pred, model_name=f"Dummy Baseline ({strategy})")


def run_full_evaluation(X_train, y_train, X_test, y_test, grid_lr, grid_mlp, grid_rf):
    """Evaluates all trained candidate models against test data using ordinal metrics."""
    metrics_dummy = evaluate_dummy_baseline(X_train, y_train, X_test, y_test, strategy="most_frequent")

    y_pred_lr = grid_lr.predict(X_test)
    y_pred_mlp = grid_mlp.predict(X_test)
    y_pred_rf = predict_ordinal_expected_value(grid_rf, X_test)

    metrics_lr = evaluate_ordinal_model(y_test, y_pred_lr, model_name="Ordinal LR")
    metrics_mlp = evaluate_ordinal_model(y_test, y_pred_mlp, model_name="Optimized MLP")
    metrics_rf = evaluate_ordinal_model(y_test, y_pred_rf, model_name="Improved RF")

    models = ['Dummy Baseline', 'Ordinal LR', 'Optimized MLP', 'Improved RF']
    metrics_list = [metrics_dummy, metrics_lr, metrics_mlp, metrics_rf]

    return models, metrics_list


def extract_ordinal_lr_weights(grid_lr, feature_names):
    """
    Extracts feature weights from Frank & Hall Ordinal Logistic Regression.
    Averages absolute beta coefficients across all binary estimators.
    """
    estimator = grid_lr.best_estimator_ if hasattr(grid_lr, 'best_estimator_') else grid_lr
    clf = estimator.named_steps['classifier']
    
    coefs = np.array([sub_model.coef_[0] for sub_model in clf.models_.values()])
    mean_abs_coefs = np.mean(np.abs(coefs), axis=0)
    
    return pd.Series(mean_abs_coefs, index=feature_names)


def extract_rf_weights(grid_rf, feature_names):
    """
    Extracts built-in Gini / MDI Feature Importances from Random Forest.
    """
    estimator = grid_rf.best_estimator_ if hasattr(grid_rf, 'best_estimator_') else grid_rf
    clf = estimator.named_steps['classifier']
    return pd.Series(clf.feature_importances_, index=feature_names)


def extract_permutation_weights(grid_model, X_test, y_test, feature_names, random_state=42):
    """
    Computes Permutation Feature Importance for model-agnostic weighting (ideal for MLP).
    """
    result = permutation_importance(
        grid_model, X_test, y_test, 
        scoring='f1_macro', n_repeats=10, 
        random_state=random_state, n_jobs=-1
    )
    importances = np.maximum(0, result.importances_mean)
    return pd.Series(importances, index=feature_names)


def compute_summary_feature_weights(grid_lr, grid_mlp, grid_rf_initial, grid_rf_improved, X_test, y_test, feature_names):
    """
    Extracts and normalizes feature importance weights across all 4 models.
    Returns a unified DataFrame normalized to sum to 1.0 per model for direct comparison.
    """
    # 1. Ordinal LR
    lr_w = extract_ordinal_lr_weights(grid_lr, feature_names)
    lr_w_norm = lr_w / lr_w.sum() if lr_w.sum() > 0 else lr_w

    # 2. Optimized MLP
    mlp_w = extract_permutation_weights(grid_mlp, X_test, y_test, feature_names)
    mlp_w_norm = mlp_w / mlp_w.sum() if mlp_w.sum() > 0 else mlp_w

    # 3. Initial RF
    rf_init_w = extract_rf_weights(grid_rf_initial, feature_names)
    rf_init_w_norm = rf_init_w / rf_init_w.sum() if rf_init_w.sum() > 0 else rf_init_w

    # 4. Improved RF
    rf_imp_w = extract_rf_weights(grid_rf_improved, feature_names)
    rf_imp_w_norm = rf_imp_w / rf_imp_w.sum() if rf_imp_w.sum() > 0 else rf_imp_w

    weights_df = pd.DataFrame({
        'Ordinal LR': lr_w_norm,
        'Optimized MLP': mlp_w_norm,
        'Initial RF': rf_init_w_norm,
        'Improved RF': rf_imp_w_norm
    }, index=feature_names)

    return weights_df