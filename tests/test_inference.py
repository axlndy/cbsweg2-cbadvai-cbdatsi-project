# tests/test_inference.py
import pandas as pd
import pytest

from src.cbdatsi.inference import (
    check_chisquare_assumptions,
    perform_chisquare_independence,
)


# ============================================================
# ASSUMPTION CHECKING
# ============================================================

def test_chisquare_assumption_met(capsys):
    """
    Verifies the branch where all expected frequencies are
    at least 5.
    """

    expected = pd.DataFrame([
        [5, 10],
        [20, 30],
    ])

    check_chisquare_assumptions(expected)

    captured = capsys.readouterr()

    assert "Assumption MET" in captured.out


def test_chisquare_assumption_violated(capsys):
    """
    Verifies the branch where at least one expected frequency
    is below 5.
    """

    expected = pd.DataFrame([
        [4, 10],
        [20, 30],
    ])

    check_chisquare_assumptions(expected)

    captured = capsys.readouterr()

    assert "Assumption VIOLATED" in captured.out


# ============================================================
# CHI-SQUARE FUNCTION
# ============================================================

def test_chisquare_returns_expected_structure():
    """
    Verifies that the Chi-Square function returns:
        - chi-square statistic
        - p-value
        - degrees of freedom
        - contingency table
    """

    df = pd.DataFrame({
        "Cluster": [
            0, 0, 0,
            1, 1, 1,
            2, 2, 2,
        ],
        "GPA": [
            1, 2, 3,
            1, 2, 3,
            1, 2, 3,
        ],
    })

    chi2, p_value, dof, table = (
        perform_chisquare_independence(df)
    )

    assert isinstance(chi2, float)
    assert isinstance(p_value, float)
    assert isinstance(dof, int)

    assert isinstance(
        table,
        pd.DataFrame
    )

    assert table.shape == (3, 3)


def test_chisquare_known_result():
    """
    Verifies the mathematical correctness of the Chi-Square
    calculation using a known contingency table.

    Expected:
        Chi-Square = 20.0
        p-value ≈ 0.0004993992
        degrees of freedom = 4
    """

    df = pd.DataFrame({
        "Cluster": (
            [0] * 60
            + [1] * 60
            + [2] * 60
        ),
        "GPA": (
            [1] * 30
            + [2] * 20
            + [3] * 10

            + [1] * 10
            + [2] * 20
            + [3] * 30

            + [1] * 20
            + [2] * 20
            + [3] * 20
        ),
    })

    chi2, p_value, dof, table = (
        perform_chisquare_independence(df)
    )

    # Mathematical expected values
    assert chi2 == pytest.approx(
        20.0,
        abs=1e-10
    )

    assert p_value == pytest.approx(
        0.0004993992273873336,
        rel=1e-10
    )

    assert dof == 4

    # Verify the actual contingency table.
    # pd.crosstab() assigns names to the index and columns.
    expected_table = pd.DataFrame(
        {
            1: [30, 10, 20],
            2: [20, 20, 20],
            3: [10, 30, 20],
        },
        index=pd.Index(
            [0, 1, 2],
            name="Cluster"
        )
    )

    expected_table.columns.name = "GPA"

    pd.testing.assert_frame_equal(
        table,
        expected_table
    )


# ============================================================
# CUSTOM COLUMN NAMES
# ============================================================

def test_chisquare_custom_column_names():
    """
    Verifies that custom target and cluster column names
    are supported.
    """

    df = pd.DataFrame({
        "StudentCluster": [
            0, 0, 0,
            1, 1, 1
        ],
        "StudentGPA": [
            1, 1, 2,
            1, 2, 2
        ],
    })

    chi2, p_value, dof, table = (
        perform_chisquare_independence(
            df,
            target_col="StudentGPA",
            cluster_col="StudentCluster",
        )
    )

    assert isinstance(chi2, float)
    assert isinstance(p_value, float)
    assert isinstance(dof, int)

    assert not table.empty
    assert table.shape == (2, 2)

    assert table.index.name == "StudentCluster"
    assert table.columns.name == "StudentGPA"


# ============================================================
# INVALID INPUTS
# ============================================================

def test_chisquare_invalid_target_column():
    """
    Verifies that a nonexistent target column raises
    an appropriate KeyError.
    """

    df = pd.DataFrame({
        "Cluster": [0, 1, 0, 1],
        "GPA": [1, 2, 2, 1],
    })

    with pytest.raises(KeyError):
        perform_chisquare_independence(
            df,
            target_col="DoesNotExist"
        )


def test_chisquare_invalid_cluster_column():
    """
    Verifies that a nonexistent cluster column raises
    an appropriate KeyError.
    """

    df = pd.DataFrame({
        "Cluster": [0, 1, 0, 1],
        "GPA": [1, 2, 2, 1],
    })

    with pytest.raises(KeyError):
        perform_chisquare_independence(
            df,
            cluster_col="DoesNotExist"
        )


# ============================================================
# EDGE CASE
# ============================================================

def test_chisquare_zero_variance():
    """
    Verifies the degenerate case where all observations have
    the same cluster and GPA category.
    """

    df = pd.DataFrame({
        "Cluster": [0, 0, 0, 0, 0],
        "GPA": [3, 3, 3, 3, 3],
    })

    chi2, p_value, dof, table = (
        perform_chisquare_independence(df)
    )

    assert chi2 == 0.0
    assert p_value == 1.0
    assert dof == 0

    assert table.loc[0, 3] == 5

    assert table.index.name == "Cluster"
    assert table.columns.name == "GPA"