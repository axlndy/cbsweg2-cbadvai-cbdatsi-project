# EduPredict

### A Multidimensional Data Mining and Machine Learning Approach to Higher Education Student Performance

Integrated Project for Term 3, AY 2025-26:

* **CBSWEG2** – Advanced Software Engineering
* **CBADVAI** – Advanced Intelligent Systems
* **CBDATSI** – Principles of Data Science

*Authors: Akisha Jeneille C. Africa, Axl Roel P. Andaya, Rienzel Kristian P. Galang*

---

## Project Overview

EduPredict is an integrated triad project that investigates the factors influencing the academic performance of higher education students through statistical analysis, data mining, and supervised machine learning.

The project utilizes the **Dataset of Factors Affecting Learning Outcomes of Students at the University of Education, Vietnam National University, Hanoi** published in *Data in Brief (2023)*. The dataset contains demographic, socioeconomic, institutional, behavioral, and academic variables collected from 2,170 students and alumni.

This repository serves as the primary development environment shared across CBSWEG2, CBADVAI, and CBDATSI. It follows a modular software engineering approach to facilitate collaborative development, automated CI/CD testing, and reproducibility.

---

## Research Questions

### CBDATSI

> *How do internal student behaviors interact with external institutional and policy support systems to characterize different profiles of academic achievement (GPA)?*

### CBADVAI

> *How do internal behavioral habits and external institutional support systems interact to influence student academic performance, and which machine learning architecture best captures these complex relationships to predict learning outcomes?*

---

## Current Project Status

### CBDATSI — **Completed**

* Structural Data Validation & Cleaning
* Feature Engineering
* Exploratory Data Analysis (EDA) & Visualizations
* K-Means Clustering
* Silhouette Evaluation
* Chi-Square Statistical Inference

### CBADVAI — **Completed**

* Data Preprocessing & Feature Selection
* Ordinal Logistic Regression
* Multi-Layer Perceptron (MLP) Neural Network
* Random Forest Classifier
* Model Evaluation Metrics
* Model Comparison & Visualization

### CBSWEG2 — **Completed**

* Modular repository architecture
* Separation of source and test modules
* Shared Pytest fixtures
* Automated CI/CD pipeline via GitHub Actions
* Unit testing across CBDATSI and CBADVAI modules
* Regression testing
* End-to-end system testing

---

## Repository Structure

```text
CBSWEG2-CBADVAI-CBDATSI-PROJECT/
│
├── .github/
│   └── workflows/
│       └── python-tests.yml          # GitHub Actions automated testing workflow
│
├── data/
│   ├── processed/
│   │   └── dataset_cache.pkl         # Cached processed dataset
│   │
│   └── raw/
│       ├── Database paper.xlsx       # Core dataset
│       └── CODEBOOK.docx             # Variable data dictionary
│
├── notebooks/
│   ├── cbadvai_project_ml.ipynb      # CBADVAI ML modeling notebook
│   └── cbdatsi_project.ipynb         # CBDATSI EDA & inference notebook
│
├── src/
│   ├── __init__.py                   # Source package initialization
│   │
│   ├── cbadvai/
│   │   ├── __init__.py               # CBADVAI package initialization
│   │   ├── metrics.py                # Model evaluation metrics
│   │   ├── models.py                 # Ordinal LR, MLP, and RF models
│   │   ├── plots.py                  # CBADVAI model visualizations
│   │   └── preprocessing.py          # Data loading and preprocessing
│   │
│   └── cbdatsi/
│       ├── __init__.py               # CBDATSI package initialization
│       ├── eda_plots.py              # Exploratory data analysis visualizations
│       ├── inference.py              # Chi-Square statistical inference
│       ├── modeling.py               # K-Means clustering and clustering plots
│       └── pipeline.py               # Dataset loading, validation, and feature engineering
│
├── tests/
│   ├── __init__.py                   # Test package initialization, if present
│   ├── conftest.py                   # Shared Pytest fixtures and mock datasets
│   │
│   ├── test_eda_plots.py             # CBDATSI EDA visualization tests
│   ├── test_inference.py             # CBDATSI Chi-Square tests
│   ├── test_metrics.py               # CBADVAI evaluation metric tests
│   ├── test_modeling.py              # CBDATSI clustering and clustering plot tests
│   ├── test_models.py                # CBADVAI model unit tests
│   ├── test_pipeline.py              # CBDATSI pipeline and data integrity tests
│   ├── test_plots.py                 # CBADVAI visualization tests
│   ├── test_preprocessing.py         # CBADVAI preprocessing tests
│   ├── test_regression.py            # Regression tests against project baseline
│   └── test_system_overall.py        # End-to-end system integration tests
│
├── .gitignore
├── README.md
└── requirements.txt                  # Python and system dependencies
```

---

## Source Modules

### CBDATSI

The CBDATSI source package contains the complete data science workflow:

* **`pipeline.py`** — Loads, caches, validates, cleans, and feature-engineers the dataset. The pipeline validates required columns, missing values, and allowed value ranges.
* **`eda_plots.py`** — Generates demographic, behavioral, socioeconomic, institutional, and mindset visualizations.
* **`modeling.py`** — Handles clustering feature selection, preprocessing, K-Means clustering, Elbow analysis, PCA visualization, cluster profiling, and silhouette evaluation.
* **`inference.py`** — Performs Chi-Square tests of independence and checks the expected-frequency assumption.

The clustering module uses 11 internal and external behavioral/institutional features for K-Means analysis.

### CBADVAI

The CBADVAI source package contains the machine learning workflow:

* **`preprocessing.py`** — Loads the dataset, performs feature extraction and binary encoding, and creates the stratified train/test split.
* **`models.py`** — Implements the Frank & Hall Ordinal Classifier and builds the Ordinal Logistic Regression, MLP, and Random Forest pipelines.
* **`metrics.py`** — Computes Macro F1, MAE, QWK, confusion matrices, and feature-weight information.
* **`plots.py`** — Generates model tuning, comparison, confusion matrix, and feature-weight visualizations.

---

## Test Suite

The project uses **Pytest** for automated testing across the CBDATSI and CBADVAI modules.

### Shared Fixtures

`tests/conftest.py` provides reusable test datasets:

* `dummy_student_data` — 5-row mock dataset representing the CBDATSI data contract.
* `synthetic_survey_df` — 100-row synthetic dataset for CBADVAI model training and evaluation.

The CBDATSI fixture contains all 22 required dataset columns with valid categorical and ordinal values.

### CBDATSI Tests

* **`test_pipeline.py`** — Tests dataset loading, caching, validation, cleaning, type conversion, and feature engineering.
* **`test_eda_plots.py`** — Tests the structure of all CBDATSI EDA visualizations, including demographics, behavioral boxplots, socioeconomic plots, institutional heatmaps, and mindset heatmaps.
* **`test_modeling.py`** — Tests clustering feature selection, preprocessing, K-Means execution, cluster summaries, silhouette evaluation, and clustering visualizations including the Elbow curve, PCA plot, and cluster profiles.
* **`test_inference.py`** — Tests Chi-Square assumptions, return structures, and mathematical correctness of the statistical inference implementation.

### CBADVAI Tests

* **`test_preprocessing.py`** — Tests the 11-feature contract, preprocessing ranges, data types, binary encoding, and data preparation behavior.
* **`test_models.py`** — Tests the Frank & Hall ordinal classifier, Ordinal Logistic Regression, MLP, and Random Forest model behavior and invariants.
* **`test_metrics.py`** — Tests metric output structure, numerical bounds, confusion matrix integrity, and edge cases.
* **`test_plots.py`** — Tests CBADVAI visualization functions for successful rendering, including model comparisons, confusion matrices, and feature-weight visualizations.

### Cross-System Tests

* **`test_regression.py`** — Verifies that the actual project dataset and major CBDATSI outputs remain consistent with established baseline expectations. The regression fixture starts from the repository's actual Excel dataset and reproduces the preprocessing pipeline.
* **`test_system_overall.py`** — Performs end-to-end integration testing across the CBDATSI and CBADVAI pipelines, including data validation, feature engineering, clustering, statistical inference, preprocessing, model training, and evaluation.

---

## Automated Testing

The project uses GitHub Actions to automatically execute the Python test suite when configured repository events occur.

The automated test suite covers:

1. **Unit Tests** — Individual functions and modules
2. **Visualization Tests** — Plot structure and rendering behavior
3. **Regression Tests** — Protection against unintended changes to established outputs
4. **System Tests** — End-to-end integration across the complete analytics and machine learning workflow

The test suite is designed to verify both individual module correctness and integration between processing, modeling, inference, and evaluation stages.

---

## Data Science Workflow

```text
Raw Dataset
     │
     ▼
Data Loading & Caching
     │
     ▼
Structural Validation
     │
     ▼
Cleaning & Typecasting
     │
     ▼
Feature Engineering
     │
     ▼
   EDA
     │
     ▼                      
     │
     ▼
K-Means Clustering
     ├──────────────────────────────┐
     ▼                              ▼
     │                              ├── Elbow Analysis
     │                              ├── PCA Visualization
     │                              ├── Cluster Profiles
     │                              └── Silhouette Evaluation
     │
     ▼
Chi-Square Statistical Inference
```

---

## Machine Learning Workflow

```text
Raw Dataset
     │
     ▼
Data Loading & Preprocessing
     │
     ▼
Feature Selection
     │
     ▼
Binary Encoding
     │
     ▼
Stratified 80/20 Train-Test Split
     │
     ▼
Model Training
     │
     ├───────────────┬───────────────┐
     ▼               ▼               ▼
Ordinal LR         MLP          Random Forest
     │               │               │
     └───────────────┴───────────────┘
                     │
                     ▼
              Model Evaluation
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Macro F1     MAE        QWK
                     │
                     ▼
              Model Comparison
```

---

## Testing Architecture

```text
                    ┌─────────────────────┐
                    │   Source Modules    │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
       ┌──────────────┐                  ┌──────────────┐
       │   CBDATSI    │                  │   CBADVAI    │
       └──────┬───────┘                  └──────┬───────┘
              │                                 │
              ▼                                 ▼
       Unit Tests                         Unit Tests
              │                                 │
              └────────────────┬────────────────┘
                               │
                               ▼
                       Regression Tests
                               │
                               ▼
                     End-to-End System Test
                               │
                               ▼
                      Automated CI/CD
```

---

## Technologies

* **Python**
* **Pandas**
* **NumPy**
* **SciPy**
* **Scikit-learn**
* **imbalanced-learn**
* **Matplotlib**
* **Seaborn**
* **Pytest**
* **GitHub Actions**
* **Jupyter Notebook**
* **Excel**

---

## Repository Goals

The repository is designed to demonstrate:

* Reproducible data science workflows
* Modular Python development
* Statistical and machine learning analysis
* Automated software testing
* Regression protection
* End-to-end integration testing
* CI/CD-based quality assurance
* Separation of analytical logic from visualization and testing
