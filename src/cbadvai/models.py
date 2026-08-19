import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
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


# HELPER FUNCTION: Expected Value Prediction for Ordinal Target
def predict_ordinal_expected_value(model, X, classes=np.array([1, 2, 3, 4, 5])):
    """
    Helper Function: Computes expected value across class probabilities to optimize MAE:
    E[y] = sum(k * P(y=k)) rounded to nearest integer bracket.
    """
    probs = model.predict_proba(X)
    
    if hasattr(model, "classes_"):
        model_classes = model.classes_
    elif hasattr(model, "named_steps") and hasattr(model.named_steps['classifier'], "classes_"):
        model_classes = model.named_steps['classifier'].classes_
    else:
        model_classes = classes

    expected_vals = np.dot(probs, model_classes)
    return np.clip(np.round(expected_vals), classes.min(), classes.max()).astype(int)


def tune_ordinal_lr(X_train, y_train, cv):
    """Executes GridSearchCV tuning for Ordinal Logistic Regression."""
    lr_pipeline = build_ordinal_lr_pipeline()
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__penalty': ['l1', 'l2']
    }
    grid_lr = GridSearchCV(lr_pipeline, param_grid, scoring='f1_macro', cv=cv, n_jobs=-1)
    grid_lr.fit(X_train, y_train)
    return grid_lr


def tune_mlp(X_train, y_train, cv):
    """
    Executes cross-validation across architectures/activations and fits
    the optimal MLP.

    In addition to the mean Macro F1, the individual fold scores are
    retained so the five-fold model-selection process can be verified
    directly by automated tests.
    """
    architectures = {
        'Shallow-Narrow (32,)': (32,),
        'Shallow-Wide (128,)': (128,),
        'Baseline (64, 32)': (64, 32),
        'Wider (128, 64)': (128, 64),
        'Deep-Narrow (32, 16, 8)': (32, 16, 8),
        'Deep-Wide (128, 64, 32)': (128, 64, 32),
        'Very Deep (128, 64, 32, 16)': (128, 64, 32, 16)
    }

    activations = ['relu', 'tanh', 'logistic']
    results = []

    for arch_name, sizes in architectures.items():
        for act in activations:

            mlp_inst = MLPClassifier(
                hidden_layer_sizes=sizes,
                activation=act,
                solver='adam',
                early_stopping=True,
                n_iter_no_change=15,
                max_iter=1500,
                random_state=42,
                alpha=0.0001
            )

            pipe = build_mlp_pipeline()
            pipe.steps[-1] = ('classifier', mlp_inst)

            cv_output = cross_validate(
                pipe,
                X_train,
                y_train,
                cv=cv,
                scoring='f1_macro',
                n_jobs=-1,
                return_train_score=False
            )

            scores = cv_output['test_score']

            row = {
                'Architecture': arch_name,
                'Activation': act,
                'Mean F1': np.mean(scores),
                'sizes': sizes
            }

            # Explicitly retain every validation-fold result.
            for fold_number, score in enumerate(scores, start=1):
                row[f'Fold_{fold_number}_F1'] = score

            results.append(row)

    results_df = pd.DataFrame(results)

    best_row = results_df.loc[
        results_df['Mean F1'].idxmax()
    ]

    best_mlp = MLPClassifier(
        hidden_layer_sizes=best_row['sizes'],
        activation=best_row['Activation'],
        solver='adam',
        early_stopping=True,
        n_iter_no_change=15,
        max_iter=1500,
        random_state=42,
        alpha=0.0001
    )

    grid_mlp = build_mlp_pipeline()
    grid_mlp.steps[-1] = (
        'classifier',
        best_mlp
    )

    # Fit the selected architecture on the complete training set.
    grid_mlp.fit(X_train, y_train)

    return (
        grid_mlp,
        results_df,
        list(architectures.keys()),
        best_row
    )


def tune_initial_rf(X_train, y_train, cv):
    """Executes initial GridSearchCV tuning for Random Forest."""
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5]
    }
    rf_pipeline = build_rf_pipeline()
    grid_rf = GridSearchCV(rf_pipeline, param_grid, scoring='f1_macro', cv=cv, n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    return grid_rf


def tune_improved_rf(X_train, y_train, cv):
    """Executes deep optimization GridSearchCV tuning for the improved Random Forest model."""
    param_grid_rf = {
        'classifier__n_estimators': [200, 300],
        'classifier__max_depth': [15, 25, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2],
        'classifier__class_weight': [None]
    }
    rf_pipeline = build_rf_pipeline()
    grid_rf = GridSearchCV(rf_pipeline, param_grid_rf, scoring='f1_macro', cv=cv, n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    return grid_rf