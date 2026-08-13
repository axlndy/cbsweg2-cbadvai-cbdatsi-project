# src/cbdatsi/inference.py
import pandas as pd
import scipy.stats as stats

def check_chisquare_assumptions(expected_freq: pd.DataFrame):
    """Checks the Chi-Square assumption that all expected counts are >= 5."""
    print("--- Assumption Check: Expected Counts ---")
    min_expected = expected_freq.min().min()
    print(f"Minimum Expected Count: {min_expected:.2f}")
    if min_expected >= 5:
        print("Conclusion: All expected counts are at least 5. Assumption MET.\n")
    else:
        print("Conclusion: Some expected counts are below 5. Assumption VIOLATED.\n")

def perform_chisquare_independence(df: pd.DataFrame, target_col: str = 'GPA', cluster_col: str = 'Cluster'):
    """Executes the Chi-Square Test of Independence as taught in class."""
    contingency_table = pd.crosstab(df[cluster_col], df[target_col])
    
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
    expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
    
    check_chisquare_assumptions(expected_df)
    
    print(f"--- Statistical Inference Results (Chi-Square Test of Independence) ---")
    print(f"Degrees of Freedom (df): {dof}")
    print(f"Chi-Square Statistic: {chi2:.4f}")
    print(f"p-value: {p_val:.4e}\n")
    
    return chi2, p_val, dof, contingency_table