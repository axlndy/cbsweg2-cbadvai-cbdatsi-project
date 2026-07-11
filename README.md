# EduPredict
### A Multidimensional Data Mining and Machine Learning Approach to Higher Education Student Performance

Integrated Project for:

- **CBSWEG2 – Advanced Software Engineering**
- **CBADVAI – Advanced Intelligent Systems**
- **CBDATSI – Principles of Data Science**

---

# Project Overview

EduPredict is an integrated triad project that investigates the factors influencing the academic performance of higher education students through statistical analysis, data mining, and supervised machine learning.

The project utilizes the **Dataset of Factors Affecting Learning Outcomes of Students at the University of Education, Vietnam National University, Hanoi** published in *Data in Brief (2025)*. The dataset contains demographic, socioeconomic, institutional, behavioral, and academic variables collected from 2,170 students and alumni.

This repository serves as the primary development repository shared across CBSWEG2, CBADVAI, and CBDATSI.

---

# Current Project Status

## CBDATSI

**Completed**

- Dataset Description
- Data Cleaning
- Exploratory Data Analysis
- Research Question Formulation

Current notebook:

```
notebooks/
└── 01_eda.ipynb
```

---

## CBADVAI

**Current Status**

The group's Initial Implementation Plan has been submitted and is currently **pending approval**.

The current proposal is to compare two supervised machine learning approaches:

- Ordinal Logistic Regression *(Proposed)*
- Multi-Layer Perceptron (MLP) Neural Network *(Proposed)*

These models may still change depending on instructor feedback.

---

## CBSWEG2

Current software engineering tasks include

- Repository organization
- Version control
- Automated testing
- GitHub Actions
- Unit testing
- System testing
- Documentation

---

# Research Question

> How do combinations of internal student behaviors (study habits, time management, and adaptation) interact with external institutional and policy support systems to characterize different profiles of academic achievement (GPA)?

---

# Dataset

**Title**

Dataset of Factors Affecting Learning Outcomes of Students at the University of Education, Vietnam National University, Hanoi

**Publication**

Tran, Nguyen & Le (2025)

**Journal**

Data in Brief

**Dataset Size**

- 2,170 observations
- 22 variables

The dataset includes

- Student demographics
- Socioeconomic indicators
- Institutional support
- Learning environment
- Study habits
- Social behaviors
- Academic performance (GPA)

---

# Planned Integrated Architecture

```
Raw Dataset
      │
      ▼
Data Validation
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
CBDATSI
Exploratory Data Analysis
      │
      ▼
CBADVAI
Supervised Machine Learning
(Proposed)
 • Ordinal Logistic Regression
 • Multi-Layer Perceptron
      │
      ▼
Performance Evaluation
      │
      ▼
Prediction & Interpretation
```

---

# Repository Structure

```
cbsweg2-cbadvai-cbdatsi-project/
│
├── .github/
│   └── workflows/
│       └── python-tests.yml      # GitHub Actions automated testing workflow
│
├── data/
│   └── Database paper.xlsx       # Raw dataset used across CBDATSI and CBADVAI
│
├── notebooks/
│   └── 01_eda.ipynb              # CBDATSI Phase 1 notebook (Dataset Description, Data Cleaning, EDA, Research Question)
│
├── src/
│   ├── __init__.py               # Makes src an importable Python package
│   └── student_eda.py            # Refactored EDA pipeline for automation and testing
│
├── tests/
│   └── test_student_eda.py       # Pytest unit tests for data preparation modules
│
├── models/                       # Future machine learning models (CBADVAI)
│
├── docs/                         # Jira documentation, reports, and project artifacts
│
├── requirements.txt              # Python dependencies
│
└── README.md                     # Project documentation
```

---

# Team Members

- **Akisha Jeneille C. Africa**
  - Developer
  - Data Analyst

- **Axl Roel P. Andaya**
  - Product Owner
  - Data Analyst

- **Rienzel Kristian P. Galang**
  - Scrum Master
  - Data Analyst

---

# Repository Notes

This repository is actively developed throughout Term 3 AY 2025–2026.

Implementation details, project architecture, and machine learning models may change as the project progresses and after consultation with the course instructors.

