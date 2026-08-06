import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, cohen_kappa_score, confusion_matrix
from sklearn.dummy import DummyClassifier
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