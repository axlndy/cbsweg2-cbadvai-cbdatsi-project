import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.cbdatsi.pipeline import (
    load_and_cache_dataset,
    validate_dataset,
    clean_and_typecast_data,
    perform_feature_engineering,
)


# ============================================================
# CONSTANTS
# ============================================================

REQUIRED_COLUMNS = [
    "Year",
    "Gender",
    "Policy_Stu",
    "Minority_Stu",
    "Poor_Stu",
    "Father_Edu",
    "Mother_Edu",
    "Father_Occupation",
    "Mother_Occupation",
    "Time_Friends",
    "Time_SocicalMedia",
    "Time_Studying",
    "GPA",
    "Adapt_Learning_Uni",
    "Study_Methods",
    "SupportOf_Uni",
    "SupportOf_Lec",
    "Facilitie_Uni",
    "Quality_Lecturer",
    "TrainingCurriculum",
    "Competitive_Class",
    "InfuenceF_Friends",
]


TYPECAST_COLUMNS = [
    "Year",
    "Gender",
    "GPA",
    "Time_Studying",
    "Time_Friends",
    "Adapt_Learning_Uni",
]


VALID_RANGES = {
    "Year": (1, 5),
    "Gender": (1, 2),
    "Policy_Stu": (1, 2),
    "Minority_Stu": (1, 2),
    "Poor_Stu": (1, 2),

    "Father_Edu": (1, 6),
    "Mother_Edu": (1, 6),

    "Father_Occupation": (1, 8),
    "Mother_Occupation": (1, 8),

    "Time_Friends": (1, 5),
    "Time_SocicalMedia": (1, 5),
    "Time_Studying": (1, 5),

    "GPA": (1, 5),

    "Adapt_Learning_Uni": (1, 5),
    "Study_Methods": (1, 5),

    "SupportOf_Uni": (1, 5),
    "SupportOf_Lec": (1, 5),
    "Facilitie_Uni": (1, 5),
    "Quality_Lecturer": (1, 5),
    "TrainingCurriculum": (1, 5),

    "Competitive_Class": (1, 5),
    "InfuenceF_Friends": (1, 5),
}


# ============================================================
# MODULE 1: LOAD & CACHE
# ============================================================

def test_load_and_cache_dataset_first_run(dummy_student_data):
    """
    Verifies that the raw Excel dataset is loaded and cached
    when no cache exists.
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        raw_path = os.path.join(temp_dir, "sample.xlsx")
        cache_path = os.path.join(
            temp_dir,
            "dataset_cache.pkl"
        )

        dummy_student_data.to_excel(
            raw_path,
            index=False
        )

        loaded_df = load_and_cache_dataset(
            raw_path,
            cache_path
        )

        assert os.path.exists(cache_path)

        pd.testing.assert_frame_equal(
            loaded_df,
            dummy_student_data
        )


def test_load_and_cache_dataset_from_cache(
    dummy_student_data
):
    """
    Verifies that an existing cache is loaded instead of
    reading the raw Excel file.
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        raw_path = os.path.join(
            temp_dir,
            "sample.xlsx"
        )

        cache_path = os.path.join(
            temp_dir,
            "dataset_cache.pkl"
        )

        # Raw file deliberately contains different data.
        pd.DataFrame({
            "Wrong": [1, 2, 3]
        }).to_excel(
            raw_path,
            index=False
        )

        # Cache contains the expected dataset.
        dummy_student_data.to_pickle(
            cache_path
        )

        loaded_df = load_and_cache_dataset(
            raw_path,
            cache_path
        )

        pd.testing.assert_frame_equal(
            loaded_df,
            dummy_student_data
        )


def test_load_and_cache_dataset_missing_file():
    """
    Verifies that a missing raw file raises FileNotFoundError
    when no cache is available.
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        raw_path = os.path.join(
            temp_dir,
            "missing.xlsx"
        )

        cache_path = os.path.join(
            temp_dir,
            "dataset_cache.pkl"
        )

        with pytest.raises(
            FileNotFoundError,
            match="Missing raw data file"
        ):
            load_and_cache_dataset(
                raw_path,
                cache_path
            )


# ============================================================
# MODULE 2: VALIDATION
# ============================================================

def test_validate_dataset_valid(
    dummy_student_data
):
    """
    Verifies that a structurally valid dataset with valid
    values passes validation.
    """

    assert validate_dataset(
        dummy_student_data
    ) is True


@pytest.mark.parametrize(
    "column",
    REQUIRED_COLUMNS
)
def test_validate_dataset_missing_column(
    dummy_student_data,
    column
):
    """
    Verifies that validation detects every required column
    when it is missing.
    """

    invalid_df = dummy_student_data.drop(
        columns=[column]
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns"
    ):
        validate_dataset(invalid_df)


@pytest.mark.parametrize(
    "missing_value",
    [np.nan, None, pd.NA]
)
def test_validate_dataset_missing_values(
    dummy_student_data,
    missing_value
):
    """
    Verifies that missing values are rejected.
    """

    invalid_df = dummy_student_data.copy()

    invalid_df.loc[
        0,
        "Gender"
    ] = missing_value

    with pytest.raises(
        ValueError,
        match="missing values"
    ):
        validate_dataset(invalid_df)


@pytest.mark.parametrize(
    "column,bad_value",
    [
        ("Year", 0),
        ("Year", 6),

        ("Gender", 0),
        ("Gender", 3),

        ("Policy_Stu", 0),
        ("Policy_Stu", 3),

        ("Minority_Stu", 0),
        ("Minority_Stu", 3),

        ("Poor_Stu", 0),
        ("Poor_Stu", 3),

        ("Father_Edu", 0),
        ("Father_Edu", 7),

        ("Mother_Edu", 0),
        ("Mother_Edu", 7),

        ("Father_Occupation", 0),
        ("Father_Occupation", 9),

        ("Mother_Occupation", 0),
        ("Mother_Occupation", 9),

        ("Time_Friends", 0),
        ("Time_Friends", 6),

        ("Time_SocicalMedia", 0),
        ("Time_SocicalMedia", 6),

        ("Time_Studying", 0),
        ("Time_Studying", 6),

        ("GPA", 0),
        ("GPA", 6),

        ("Adapt_Learning_Uni", 0),
        ("Adapt_Learning_Uni", 6),

        ("Study_Methods", 0),
        ("Study_Methods", 6),

        ("SupportOf_Uni", 0),
        ("SupportOf_Uni", 6),

        ("SupportOf_Lec", 0),
        ("SupportOf_Lec", 6),

        ("Facilitie_Uni", 0),
        ("Facilitie_Uni", 6),

        ("Quality_Lecturer", 0),
        ("Quality_Lecturer", 6),

        ("TrainingCurriculum", 0),
        ("TrainingCurriculum", 6),

        ("Competitive_Class", 0),
        ("Competitive_Class", 6),

        ("InfuenceF_Friends", 0),
        ("InfuenceF_Friends", 6),
    ]
)
def test_validate_dataset_invalid_range(
    dummy_student_data,
    column,
    bad_value
):
    """
    Verifies that values outside the documented coding range
    are rejected.
    """

    invalid_df = dummy_student_data.copy()

    invalid_df.loc[
        0,
        column
    ] = bad_value

    with pytest.raises(
        ValueError,
        match=f"Invalid values in {column}"
    ):
        validate_dataset(invalid_df)


def test_validate_dataset_non_numeric_value(
    dummy_student_data
):
    """
    Verifies that a non-numeric value is rejected by the
    range validation.
    """

    invalid_df = dummy_student_data.copy()

    invalid_df["GPA"] = (
        invalid_df["GPA"].astype(object)
    )

    invalid_df.loc[0, "GPA"] = "High"

    with pytest.raises(
        ValueError,
        match="Invalid data type in GPA"
    ):
        validate_dataset(invalid_df)


def test_validate_dataset_empty_dataframe():
    """
    Verifies that an empty DataFrame fails because all
    required columns are missing.
    """

    with pytest.raises(
        ValueError,
        match="Missing required columns"
    ):
        validate_dataset(
            pd.DataFrame()
        )


# ============================================================
# MODULE 3: CLEAN & TYPECAST
# ============================================================

@pytest.mark.parametrize(
    "column",
    TYPECAST_COLUMNS
)
def test_clean_and_typecast_target_columns(
    dummy_student_data,
    column
):
    """
    Verifies that every target column is converted to int64.
    """

    float_df = dummy_student_data.astype(float)

    result = clean_and_typecast_data(
        float_df
    )

    assert result[column].dtype == np.int64


def test_clean_and_typecast_preserves_values(
    dummy_student_data
):
    """
    Verifies that typecasting changes the data type without
    changing the underlying values.
    """

    float_df = dummy_student_data.astype(float)

    result = clean_and_typecast_data(
        float_df
    )

    for column in TYPECAST_COLUMNS:
        pd.testing.assert_series_equal(
            result[column],
            dummy_student_data[column],
            check_dtype=False
        )


def test_clean_and_typecast_invalid_string(
    dummy_student_data
):
    """
    Verifies that an invalid string cannot be converted
    to an integer.
    """

    corrupted_df = dummy_student_data.copy()

    corrupted_df["GPA"] = (
        corrupted_df["GPA"].astype(object)
    )

    corrupted_df.loc[0, "GPA"] = "High"

    with pytest.raises(
        (TypeError, ValueError)
    ):
        clean_and_typecast_data(
            corrupted_df
        )


def test_clean_and_typecast_does_not_modify_original(
    dummy_student_data
):
    """
    Verifies that typecasting does not modify the source
    DataFrame in place.
    """

    original = dummy_student_data.copy()

    clean_and_typecast_data(
        dummy_student_data
    )

    pd.testing.assert_frame_equal(
        dummy_student_data,
        original
    )


# ============================================================
# MODULE 4: FEATURE ENGINEERING
# ============================================================

@pytest.mark.parametrize(
    "value,expected",
    [
        (1, "Poor"),
        (2, "Average"),
        (3, "Fair"),
        (4, "Good"),
        (5, "Excellent"),
    ]
)
def test_gpa_mapping(
    dummy_student_data,
    value,
    expected
):
    """
    Verifies every GPA-to-label mapping.
    """

    df = dummy_student_data.copy()
    df["GPA"] = value

    result = perform_feature_engineering(df)

    assert result["GPA_Label"].iloc[0] == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (1, "First-year"),
        (2, "Second-year"),
        (3, "Third-year"),
        (4, "Fourth-year"),
        (5, "Graduated"),
    ]
)
def test_year_mapping(
    dummy_student_data,
    value,
    expected
):
    """
    Verifies every Year-to-label mapping.
    """

    df = dummy_student_data.copy()
    df["Year"] = value

    result = perform_feature_engineering(df)

    assert result["Year_Label"].iloc[0] == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (1, "Male"),
        (2, "Female"),
    ]
)
def test_gender_mapping(
    dummy_student_data,
    value,
    expected
):
    """
    Verifies every Gender-to-label mapping.
    """

    df = dummy_student_data.copy()
    df["Gender"] = value

    result = perform_feature_engineering(df)

    assert result["Gender_Label"].iloc[0] == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (1, "Yes (Poor)"),
        (2, "No (Not Poor)"),
    ]
)
def test_poor_student_mapping(
    dummy_student_data,
    value,
    expected
):
    """
    Verifies every Poor_Stu-to-label mapping.
    """

    df = dummy_student_data.copy()
    df["Poor_Stu"] = value

    result = perform_feature_engineering(df)

    assert result["Poor_Stu_Label"].iloc[0] == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (1, "Yes (Supported)"),
        (2, "No (Not Supported)"),
    ]
)
def test_policy_student_mapping(
    dummy_student_data,
    value,
    expected
):
    """
    Verifies every Policy_Stu-to-label mapping.
    """

    df = dummy_student_data.copy()
    df["Policy_Stu"] = value

    result = perform_feature_engineering(df)

    assert result["Policy_Stu_Label"].iloc[0] == expected


def test_unmapped_gpa_returns_nan(
    dummy_student_data
):
    """
    Verifies that an unmapped GPA code produces NaN instead
    of an incorrect label.
    """

    df = dummy_student_data.copy()
    df["GPA"] = 99

    result = perform_feature_engineering(df)

    assert pd.isna(
        result["GPA_Label"].iloc[0]
    )


def test_feature_engineering_does_not_modify_original(
    dummy_student_data
):
    """
    Verifies that feature engineering does not modify the
    original DataFrame in place.
    """

    original = dummy_student_data.copy()

    perform_feature_engineering(
        dummy_student_data
    )

    pd.testing.assert_frame_equal(
        dummy_student_data,
        original
    )