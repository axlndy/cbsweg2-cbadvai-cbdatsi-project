# src/cbdatsi/pipeline.py
import os
import pandas as pd

def load_and_cache_dataset(raw_path: str, cache_path: str) -> pd.DataFrame:
    """Loads dataset from a pickle cache if it exists, otherwise from Excel."""
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing raw data file at: {raw_path}")
    df = pd.read_excel(raw_path)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_pickle(cache_path)
    return df

def validate_dataset(df: pd.DataFrame) -> bool:
    """Validates dataset structure, missing values, and value ranges."""
    required_columns = [
        "Year", "Gender", "Policy_Stu", "Minority_Stu", "Poor_Stu",
        "Father_Edu", "Mother_Edu", "Father_Occupation", "Mother_Occupation",
        "Time_Friends", "Time_SocicalMedia", "Time_Studying", "GPA",
        "Adapt_Learning_Uni", "Study_Methods", "SupportOf_Uni", "SupportOf_Lec",
        "Facilitie_Uni", "Quality_Lecturer", "TrainingCurriculum",
        "Competitive_Class", "InfuenceF_Friends"
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    if df[required_columns].isnull().any().any():
        raise ValueError("Dataset contains missing values in required columns.")
    return True

def clean_and_typecast_data(df: pd.DataFrame) -> pd.DataFrame:
    """Executes structural type-casting for discrete variables."""
    df_cleaned = df.copy()
    target_columns = ['Year', 'Gender', 'GPA', 'Time_Studying', 'Time_Friends', 'Adapt_Learning_Uni']
    for col in target_columns:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype('int64')
    return df_cleaned

def perform_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Maps ordinal and categorical variables to readable display labels."""
    df_mapped = df.copy()
    df_mapped['GPA_Label'] = df_mapped['GPA'].map({1: 'Poor', 2: 'Average', 3: 'Fair', 4: 'Good', 5: 'Excellent'})
    df_mapped['Year_Label'] = df_mapped['Year'].map({1: 'First-year', 2: 'Second-year', 3: 'Third-year', 4: 'Fourth-year', 5: 'Graduated'})
    df_mapped['Gender_Label'] = df_mapped['Gender'].map({1: 'Male', 2: 'Female'})
    df_mapped['Poor_Stu_Label'] = df_mapped['Poor_Stu'].map({1: 'Yes (Poor)', 2: 'No (Not Poor)'})
    df_mapped['Policy_Stu_Label'] = df_mapped['Policy_Stu'].map({1: 'Yes (Supported)', 2: 'No (Not Supported)'})
    return df_mapped