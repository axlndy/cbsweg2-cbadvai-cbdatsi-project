# src/cbdatsi/inference.py
import pandas as pd
from scipy.stats import f_oneway, shapiro, levene

def run_anova_inference(df: pd.DataFrame, cluster_col: str = 'Cluster', target_col: str = 'GPA') -> dict:
    """Performs normality checks, variance homogeneity testing, and One-Way ANOVA."""
    clusters = sorted(df[cluster_col].unique())
    group_data = [df[df[cluster_col] == c][target_col].dropna() for c in clusters]
    
    results = {}
    
    # 1. Normality check (Shapiro-Wilk) per cluster group
    normality_results = {}
    for c, data in zip(clusters, group_data):
        stat, pval = shapiro(data)
        normality_results[f"Cluster_{c}"] = {'statistic': stat, 'p_value': pval}
    results['normality'] = normality_results
    
    # 2. Homogeneity of variance (Levene's test)
    stat_lev, p_lev = levene(*group_data)
    results['levene'] = {'statistic': stat_lev, 'p_value': p_lev}
    
    # 3. One-Way ANOVA test
    stat_anova, p_anova = f_oneway(*group_data)
    results['anova'] = {'f_statistic': stat_anova, 'p_value': p_anova}
    
    return results