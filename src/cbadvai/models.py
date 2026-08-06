import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

class FrankHallOrdinalClassifier(BaseEstimator, ClassifierMixin):
    """
    Ordinal Classifier implementing the Frank & Hall (2001) method.
    Transforms a k-class ordinal problem into k-1 binary classification problems.
    """
    def __init__(self, C=1.0, penalty='l2', solver='saga', max_iter=1000, random_state=42):
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        self.models_ = {}
        
        for i in range(len(self.classes_) - 1):
            threshold = self.classes_[i]
            y_binary = (y > threshold).astype(int)
            
            model = LogisticRegression(
                C=self.C, 
                penalty=self.penalty, 
                solver=self.solver, 
                max_iter=self.max_iter,
                random_state=self.random_state
            )
            model.fit(X, y_binary)
            self.models_[threshold] = model
            
        return self

    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        probas_greater_than = {
            thresh: self.models_[thresh].predict_proba(X)[:, 1] 
            for thresh in self.models_
        }
        
        probas = np.zeros((n_samples, n_classes))
        
        for i, c in enumerate(self.classes_):
            if i == 0:
                probas[:, i] = 1.0 - probas_greater_than[self.classes_[i]]
            elif i == n_classes - 1:
                probas[:, i] = probas_greater_than[self.classes_[i-1]]
            else:
                probas[:, i] = (probas_greater_than[self.classes_[i-1]] - 
                                probas_greater_than[self.classes_[i]])
        
        probas = np.clip(probas, 0, 1)
        row_sums = probas.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1 
        
        return probas / row_sums

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


def build_ordinal_lr_pipeline(random_state=42):
    """Pipeline builder for Frank & Hall Ordinal Logistic Regression + SMOTE."""
    ordinal_lr = FrankHallOrdinalClassifier(solver='saga', random_state=random_state)
    return ImbPipeline([
        ('smote', SMOTE(random_state=random_state)),
        ('classifier', ordinal_lr)
    ])

def build_mlp_pipeline(random_state=42):
    """Pipeline builder for MLP Classifier + SMOTE."""
    mlp = MLPClassifier(solver='adam', early_stopping=True, n_iter_no_change=15, max_iter=1500, random_state=random_state)
    return ImbPipeline([
        ('smote', SMOTE(random_state=random_state)),
        ('classifier', mlp)
    ])

def build_rf_pipeline(random_state=42):
    """Pipeline builder for Random Forest Classifier + SMOTE."""
    rf = RandomForestClassifier(random_state=random_state)
    return ImbPipeline([
        ('smote', SMOTE(random_state=random_state)),
        ('classifier', rf)
    ])

def predict_ordinal_expected_value(model, X, classes=np.array([1, 2, 3, 4, 5])):
    """
    Computes expected value across class probabilities to directly optimize MAE:
    E[y] = sum(k * P(y=k)) rounded to nearest integer bracket.
    """
    # Obtain class probabilities (shape: n_samples x n_classes)
    probs = model.predict_proba(X)
    
    # Check if model has fewer classes than dataset
    if hasattr(model, "classes_"):
        model_classes = model.classes_
    elif hasattr(model, "named_steps") and hasattr(model.named_steps['classifier'], "classes_"):
        model_classes = model.named_steps['classifier'].classes_
    else:
        model_classes = classes

    # Calculate probability-weighted sum
    expected_vals = np.dot(probs, model_classes)
    
    # Round and clip to valid GPA range [1, 5]
    return np.clip(np.round(expected_vals), classes.min(), classes.max()).astype(int)