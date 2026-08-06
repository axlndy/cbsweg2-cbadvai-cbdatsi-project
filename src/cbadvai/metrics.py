import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, cohen_kappa_score, confusion_matrix
from sklearn.dummy import DummyClassifier

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