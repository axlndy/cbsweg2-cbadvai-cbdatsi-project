import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, mean_absolute_error

class FrankHallOrdinalClassifier(BaseEstimator, ClassifierMixin):
    """
    Ordinal Classifier implementing the Frank & Hall method.
    Transforms a k-class ordinal problem into k-1 binary classification problems.
    """
    def __init__(self, C=1.0, penalty='l2', solver='saga', max_iter=1000):
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.models = {}
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        
        # Create k-1 binary classifiers
        for i in range(len(self.classes_) - 1):
            threshold = self.classes_[i]
            # Binary target: 1 if y > threshold, 0 otherwise
            y_binary = (y > threshold).astype(int)
            
            model = LogisticRegression(
                C=self.C, 
                penalty=self.penalty, 
                solver=self.solver, 
                max_iter=self.max_iter,
                random_state=42
            )
            model.fit(X, y_binary)
            self.models[threshold] = model
            
        return self

    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Store P(y > k) for all thresholds
        probas_greater_than = {
            thresh: self.models[thresh].predict_proba(X)[:, 1] 
            for thresh in self.models
        }
        
        probas = np.zeros((n_samples, n_classes))
        
        for i, c in enumerate(self.classes_):
            if i == 0:
                # P(y = 1) = 1 - P(y > 1)
                probas[:, i] = 1.0 - probas_greater_than[self.classes_[i]]
            elif i == n_classes - 1:
                # P(y = max_class) = P(y > max_class - 1)
                probas[:, i] = probas_greater_than[self.classes_[i-1]]
            else:
                # P(y = k) = P(y > k-1) - P(y > k)
                probas[:, i] = (probas_greater_than[self.classes_[i-1]] - 
                              probas_greater_than[self.classes_[i]])
        
        # Clip to handle minor float precision anomalies and normalize
        probas = np.clip(probas, 0, 1)
        row_sums = probas.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1 
        probas = probas / row_sums
        
        return probas

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


def evaluate_ordinal_model(y_true, y_pred, model_name="Model"):
    """
    Computes and prints the required performance metrics for ordinal classification.
    """
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    mae = mean_absolute_error(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    print(f"--- Evaluation Metrics: {model_name} ---")
    print(f"Macro-Averaged F1-Score: {macro_f1:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}\n")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("-" * 40)
    
    return {
        "Macro_F1": macro_f1,
        "MAE": mae,
        "Confusion_Matrix": conf_matrix
    }