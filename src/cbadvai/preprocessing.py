import os
import pandas as pd
from sklearn.model_selection import train_test_split

SELECTED_FEATURES = [
    'Study_Methods', 'Time_Studying', 'Time_Friends', 'Time_SocicalMedia', 'Adapt_Learning_Uni',
    'Policy_Stu', 'SupportOf_Uni', 'SupportOf_Lec', 'Facilitie_Uni', 'Quality_Lecturer', 'TrainingCurriculum'
]

def load_and_preprocess_data(
    file_path: str = '../data/raw/Database paper.xlsx', 
    target_col: str = 'GPA',
    use_cache: bool = False,
    cache_path: str = '../data/processed/dataset_cache.pkl'
):
    """
    Loads survey dataset from data/raw/Database paper.xlsx or cached pickle.
    Handles duplicates, target/binary mapping, and feature extraction.
    """
    if use_cache and os.path.exists(cache_path):
        print(f"Loading cached dataset from: {cache_path}")
        df_cached = pd.read_pickle(cache_path)
        return df_cached[SELECTED_FEATURES], df_cached[target_col]

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at path: {file_path}")

    print(f"Loading raw dataset from: {file_path}")
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    if 'Policy_Stu' in df.columns:
        df['Policy_Stu'] = df['Policy_Stu'].map({1: 1, 2: 0})

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_pickle(cache_path)

    X = df[SELECTED_FEATURES]
    y = df[target_col]

    return X, y

def get_train_test_split(X, y, test_size=0.20, random_state=42):
    """
    Creates an 80/20 Stratified Holdout Split to isolate testing data.
    """
    return train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )
