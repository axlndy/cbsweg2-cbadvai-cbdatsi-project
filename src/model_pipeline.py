# ==========================================
# Authors: Africa, Akisha Jeneille; Andaya, Axl Roel; Galang, Rienzel Kristian
# Project: CBSWEG2 MCO4 - Student ML Pipelines
# ==========================================

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

def train_logistic_regression(X_train, y_train):
    """
    Initializes and trains a baseline Logistic Regression model 
    for multiclass/ordinal classification.
    """
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_neural_network(X_train, y_train):
    """
    Initializes and trains a Multilayer Perceptron (MLP) Neural Network.
    """
    # 100 hidden nodes is a standard baseline starting point
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def generate_predictions(model, X_test):
    """
    Generates predictions from a trained model.
    """
    return model.predict(X_test)