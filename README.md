# Integrated Project (Term 3, AY 2025-2026)

## Project Overview
This repository contains the integrated project for **CBSWEG2**, **CBADVAI**, and **CBDATSI**. The specifics of this project are yet to be finalized. 

## Team Members
* Akisha Jeneille C. Africa
* Axl Roel P. Andaya
* Rienzel Kristian P. Galang

## Directory Structure
Below is our tentative directory structure for the integrated project:

```text
project-repo/
├── .github/
│   └── workflows/
│       └── python-tests.yml   # GitHub Actions automated testing workflow
├── src/                           # Source code for the main application
│      ├── __init__.py            # Makes the directory an importable package
│      └── student_eda.py         # Main data processing pipeline and visualization script                 
├── data/
│      └── Database paper.xlsx        # Raw data file used for CBDATSI and CBADVAI
├── tests/
│      └── test_student_eda.py    # Pytest unit tests tracking data cleaning and mapping logic
├── models/                  # AI/ML models and training scripts
├── docs/                    # Project documentation and Jira exports
├── requirements.txt           # Dependency list
└── README.md
