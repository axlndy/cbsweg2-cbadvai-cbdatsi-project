# src/student_eda.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_cache_dataset(raw_path: str, cache_path: str) -> pd.DataFrame:
    """Loads dataset from a pickle cache if it exists, otherwise from Excel."""
    if os.path.exists(cache_path):
        print("Loading data from cache...")
        df = pd.read_pickle(cache_path)
    else:
        print("Loading raw Excel file and building cache...")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Missing raw data file at: {raw_path}")
        df = pd.read_excel(raw_path)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_pickle(cache_path)
    print(f"Data has been loaded successfully! Shape: {df.shape}")
    return df

def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Validates the dataset before preprocessing.

    Validation checks:
    - Required columns are present.
    - Required columns do not contain missing values.
    - Encoded survey variables contain only valid codes.
    - Reports duplicate rows for awareness.

    Returns:
        bool: True if the dataset passes validation.

    Raises:
        ValueError:
            - Missing required columns.
            - Missing values in required columns.
            - Invalid encoded survey values.
    """

    required_columns = [
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
        "InfuenceF_Friends"
    ]

    # ==========================================
    # Check required columns
    # ==========================================

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_columns)}"
        )

    # ==========================================
    # Check missing values
    # ==========================================

    if df[required_columns].isnull().any().any():
        raise ValueError(
            "Dataset contains missing values in one or more required columns."
        )

    # ==========================================
    # Validate encoded survey values
    # ==========================================

    valid_ranges = {
        "Year": {3, 4, 5},
        "Gender": {1, 2},
        "Policy_Stu": {1, 2},
        "Minority_Stu": {1, 2},
        "Poor_Stu": {1, 2},
        "Father_Edu": {1, 2, 3, 4, 5, 6},
        "Mother_Edu": {1, 2, 3, 4, 5, 6},
        "Father_Occupation": {1, 2, 3, 4, 5},
        "Mother_Occupation": {1, 2, 3, 4, 5},
        "Time_Friends": {1, 2, 3, 4, 5},
        "Time_SocicalMedia": {1, 2, 3, 4, 5},
        "Time_Studying": {1, 2, 3, 4, 5},
        "GPA": {1, 2, 3, 4, 5},
        "Adapt_Learning_Uni": {1, 2, 3, 4, 5},
        "Study_Methods": {1, 2, 3, 4, 5},
        "SupportOf_Uni": {1, 2, 3, 4, 5},
        "SupportOf_Lec": {1, 2, 3, 4, 5},
        "Facilitie_Uni": {1, 2, 3, 4, 5},
        "Quality_Lecturer": {1, 2, 3, 4, 5},
        "TrainingCurriculum": {1, 2, 3, 4, 5},
        "Competitive_Class": {1, 2, 3, 4, 5},
        "InfuenceF_Friends": {1, 2, 3, 4, 5}
    }

    for column, valid_values in valid_ranges.items():

        invalid_values = (
            set(df[column].dropna().unique())
            - valid_values
        )

        if invalid_values:
            raise ValueError(
                f"Invalid value(s) found in '{column}': "
                f"{sorted(invalid_values)}. "
                f"Expected values: {sorted(valid_values)}"
            )

    # ==========================================
    # Report duplicate rows
    # ==========================================

    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        print(f"Warning: {duplicate_count} duplicate rows detected.")
    else:
        print("No duplicate rows detected.")

    print("Dataset validation passed.")

    return True

def clean_and_typecast_data(df: pd.DataFrame) -> pd.DataFrame:
    """Executes structural integrity fixes via specific data type casting."""
    df_cleaned = df.copy()
    target_columns = ['Year', 'Gender', 'GPA', 'Time_Studying', 'Time_Friends', 'Adapt_Learning_Uni']
    for col in target_columns:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype('int64')
    return df_cleaned

def perform_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Maps continuous/ordinal categorical columns to distinct textual visualizations labels."""
    df_mapped = df.copy()

    # Target and Demographic Mapping
    df_mapped['GPA_Label'] = df_mapped['GPA'].map({1: 'Poor', 2: 'Average', 3: 'Fair', 4: 'Good', 5: 'Excellent'})
    df_mapped['Year_Label'] = df_mapped['Year'].map({1: 'First-year', 2: 'Second-year', 3: 'Third-year', 4: 'Fourth-year', 5: 'Graduated'})
    df_mapped['Gender_Label'] = df_mapped['Gender'].map({1: 'Male', 2: 'Female'})

    # Socioeconomic Mapping
    df_mapped['Poor_Stu_Label'] = df_mapped['Poor_Stu'].map({1: 'Yes (Poor)', 2: 'No (Not Poor)'})
    df_mapped['Policy_Stu_Label'] = df_mapped['Policy_Stu'].map({1: 'Yes (Supported)', 2: 'No (Not Supported)'})
    df_mapped['Minority_Stu_Label'] = df_mapped['Minority_Stu'].map({1: 'Yes (Minority)', 2: 'No (Non-Minority)'})

    return df_mapped

# ==========================================
#         TERMINAL OUTPUT FUNCTIONS
# ==========================================

def print_section_3_1_distributions(df: pd.DataFrame) -> None:
    """Prints baseline demographic frequencies (Section 3.1)."""
    print("\n==============================================")
    print("3.1 DEMOGRAPHIC AND TARGET BASELINE OVERVIEW")
    print("==============================================")
    for col, name in [('Year_Label', 'Year Level'), ('Gender_Label', 'Gender'), ('GPA_Label', 'GPA Bracket')]:
        if col in df.columns:
            counts = df[col].value_counts()
            percs = df[col].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            summary = pd.DataFrame({'Count': counts, 'Percentage': percs})
            print(f"\nDistribution of {name}:")
            print(summary.to_string())

def print_section_3_2_behavioral_data(df: pd.DataFrame) -> None:
    """Prints central tendency and Spearman Rank Correlation for Time Allocation (Section 3.2)."""
    print("\n==============================================")
    print("3.2 STUDENT BEHAVIORAL TIME ALLOCATION TRENDS")
    print("==============================================")
    eda1_cols = ['GPA', 'Time_Studying', 'Time_SocicalMedia']
    
    print("\n1. CENTRAL TENDENCY AND SPREAD SUMMARY")
    stats1 = df[eda1_cols].agg(['median']).T
    stats1['IQR'] = df[eda1_cols].quantile(0.75) - df[eda1_cols].quantile(0.25)
    print(stats1.round(2).to_string())
    
    print("\n2. SPEARMAN RANK CORRELATION MATRIX")
    correlation_matrix = df[eda1_cols].corr(method='spearman')
    print(correlation_matrix.round(3).to_string())

def print_section_3_3_socioeconomic_data(df: pd.DataFrame) -> None:
    """Prints group-level statistics and Spearman Correlation for Socioeconomic analysis (Section 3.3)."""
    print("\n==============================================")
    print("3.3 SOCIOECONOMIC IMPACT ON ACADEMIC PERFORMANCE")
    print("==============================================")
    for col, name in [('Poor_Stu', 'Household Wealth'), ('Policy_Stu', 'Policy Support')]:
        label_col = col + '_Label'
        grouped_stats = df.groupby(label_col)['GPA'].agg(
            Median='median',
            IQR=lambda x: x.quantile(0.75) - x.quantile(0.25)
        )
        spearman_corr = df[col].corr(df['GPA'], method='spearman')
        
        print(f"\nGPA Center & Spread by {name}:")
        print(grouped_stats.to_string())
        print(f"Spearman Rank Correlation with GPA: {spearman_corr:.3f}")

def print_section_3_4_institutional_data(df: pd.DataFrame) -> None:
    """Prints sorted Spearman Rank Correlations for Institutional framework variables (Section 3.4)."""
    print("\n==============================================")
    print("3.4 INSTITUTIONAL INFLUENCE ON ACADEMIC OUTCOMES")
    print("==============================================")
    inst_cols = ['SupportOf_Uni', 'SupportOf_Lec', 'Facilitie_Uni', 'Quality_Lecturer', 'TrainingCurriculum', 'GPA']
    inst_corr = df[inst_cols].corr(method='spearman')
    gpa_corr_focused = inst_corr[['GPA']].drop('GPA').sort_values(by='GPA', ascending=False)
    print("\nSorted Institutional Factors Correlation to GPA:")
    print(gpa_corr_focused.round(3).to_string())

def print_section_3_5_mindset_data(df: pd.DataFrame) -> None:
    """Prints sorted Spearman Rank Correlations for internal mindset vs external variables (Section 3.5)."""
    print("\n==============================================")
    print("3.5 INTERNAL MINDSET VS. EXTERNAL PRESSURES")
    print("==============================================")
    mindset_cols = ['Study_Methods', 'Adapt_Learning_Uni', 'Competitive_Class', 'InfuenceF_Friends', 'GPA']
    mindset_corr = df[mindset_cols].corr(method='spearman')
    mindset_focused = mindset_corr[['GPA']].drop('GPA').sort_values(by='GPA', ascending=False)
    print("\nSorted Mindset vs External Pressure Correlation to GPA:")
    print(mindset_focused.round(3).to_string())

# ==========================================
#          PLOT GENERATION FUNCTIONS
# ==========================================

def save_demographic_plots(df: pd.DataFrame, base_dir: str) -> None:
    """Generates and saves the baseline demographic plots (Section 3.1)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    def plot_univariate_bar(ax, series, title, order=None, color='#2F5597'):
        counts = series.value_counts()
        if order:
            counts = counts.reindex(order).fillna(0)
        ax.bar(counts.index.astype(str), counts.values, color=color, edgecolor='black', alpha=0.85)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Students')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.tick_params(axis='x', rotation=15)

    year_order = ['First-year', 'Second-year', 'Third-year', 'Fourth-year', 'Graduated']
    plot_univariate_bar(axes[0], df['Year_Label'], 'Distribution by Year Level', order=year_order, color='#2F5597')
    plot_univariate_bar(axes[1], df['Gender_Label'], 'Distribution by Gender', color='#8FAADC')
    
    gpa_order = ['Poor', 'Average', 'Fair', 'Good', 'Excellent']
    plot_univariate_bar(axes[2], df['GPA_Label'], 'Baseline GPA Distribution', order=gpa_order, color='#4472C4')

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "01_demographics.png"))
    plt.close()
    print("Saved visualization: 01_demographics.png")

def save_behavioral_plots(df: pd.DataFrame, base_dir: str) -> None:
    """Generates and saves the Time Allocation Behavioral plots (Section 3.2)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors_study = ['#E2EFDA', '#C6E0B4', '#A9D08E', '#70AD47', '#385723']
    colors_social = ['#FFF2CC', '#FFE699', '#F8CBAD', '#F4B084', '#C65911']

    # Study Time
    ct_study = pd.crosstab(df['GPA'], df['Time_Studying'], normalize='index') * 100
    ct_study.plot(kind='bar', ax=axes[0], color=colors_study, edgecolor='black', width=0.8)
    axes[0].set_title('Proportional Study Time by GPA Bracket', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('GPA Bracket (1 = Poor, 5 = Excellent)')
    axes[0].set_ylabel('Percentage of Students (%)')
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # Social Media Time
    ct_social = pd.crosstab(df['GPA'], df['Time_SocicalMedia'], normalize='index') * 100
    ct_social.plot(kind='bar', ax=axes[1], color=colors_social, edgecolor='black', width=0.8)
    axes[1].set_title('Proportional Social Media Time by GPA Bracket', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('GPA Bracket (1 = Poor, 5 = Excellent)')
    axes[1].set_ylabel('Percentage of Students (%)')
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "02_behavioral_trends.png"))
    plt.close()
    print("Saved visualization: 02_behavioral_trends.png")

def save_socioeconomic_plots(df: pd.DataFrame, base_dir: str) -> None:
    """Generates and saves the Socioeconomic Impact plots (Section 3.3)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    gpa_order = ['Poor', 'Average', 'Fair', 'Good', 'Excellent']
    dark_gpa_colors = ['#8C2D19', '#B37D14', '#4A5568', '#1F4E79', '#134E3A']

    # Wealth vs GPA
    ct_wealth = pd.crosstab(df['Poor_Stu_Label'], df['GPA_Label'], normalize='index') * 100
    ct_wealth = ct_wealth.reindex(columns=gpa_order)
    ct_wealth.plot(kind='bar', ax=axes[0], color=dark_gpa_colors, edgecolor='black', width=0.8)
    axes[0].set_title('Proportional GPA Breakdown by Household Wealth', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Percentage of Cohort (%)')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

    # Policy vs GPA
    ct_policy = pd.crosstab(df['Policy_Stu_Label'], df['GPA_Label'], normalize='index') * 100
    ct_policy = ct_policy.reindex(columns=gpa_order)
    ct_policy.plot(kind='bar', ax=axes[1], color=dark_gpa_colors, edgecolor='black', width=0.8)
    axes[1].set_title('Proportional GPA Breakdown by Policy Support', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Percentage of Cohort (%)')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "03_socioeconomic_impact.png"))
    plt.close()
    print("Saved visualization: 03_socioeconomic_impact.png")

def save_institutional_plots(df: pd.DataFrame, base_dir: str) -> None:
    """Generates and saves the Institutional Influence correlation heatmap (Section 3.4)."""
    inst_cols = ['SupportOf_Uni', 'SupportOf_Lec', 'Facilitie_Uni', 'Quality_Lecturer', 'TrainingCurriculum', 'GPA']
    inst_corr = df[inst_cols].corr(method='spearman')
    gpa_corr_focused = inst_corr[['GPA']].drop('GPA').sort_values(by='GPA', ascending=False)

    plt.figure(figsize=(6, 5))
    sns.heatmap(gpa_corr_focused, annot=True, cmap='coolwarm', fmt=".3f", vmin=-0.2, vmax=0.2, linewidths=1.5, linecolor='black')
    plt.title('Institutional Factors vs. Student GPA', fontsize=11, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "04_institutional_influences.png"))
    plt.close()
    print("Saved visualization: 04_institutional_influences.png")

def save_mindset_plots(df: pd.DataFrame, base_dir: str) -> None:
    """Generates and saves the Mindset vs External Pressure heatmap (Section 3.5)."""
    mindset_cols = ['Study_Methods', 'Adapt_Learning_Uni', 'Competitive_Class', 'InfuenceF_Friends', 'GPA']
    mindset_corr = df[mindset_cols].corr(method='spearman')
    mindset_focused = mindset_corr[['GPA']].drop('GPA').sort_values(by='GPA', ascending=False)

    plt.figure(figsize=(6, 5))
    sns.heatmap(mindset_focused, annot=True, cmap='viridis', fmt=".3f", vmin=-0.2, vmax=0.2, linewidths=1.5, linecolor='black')
    plt.title('Internal Mindset vs. External Pressures on GPA', fontsize=11, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "05_mindset_vs_pressures.png"))
    plt.close()
    print("Saved visualization: 05_mindset_vs_pressures.png")

# ==========================================
#               MAIN PIPELINE
# ==========================================

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    RAW_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

    RAW_DATA = os.path.join(RAW_DIR, "Database paper.xlsx")
    CACHE_DATA = os.path.join(PROCESSED_DIR, "dataset_cache.pkl")

    try:
        # 1. Load dataset
        raw_df = load_and_cache_dataset(RAW_DATA, CACHE_DATA)

        # 2. Validate dataset
        validate_dataset(raw_df)

        # 3. Clean dataset
        cleaned_df = clean_and_typecast_data(raw_df)

        # 4. Feature engineering
        final_df = perform_feature_engineering(cleaned_df)

        # 5. Statistical tables
        print_section_3_1_distributions(final_df)
        print_section_3_2_behavioral_data(raw_df)
        print_section_3_3_socioeconomic_data(final_df)
        print_section_3_4_institutional_data(raw_df)
        print_section_3_5_mindset_data(raw_df)

        # 6. Visualizations
        print("\n" + "=" * 46)
        print("GENERATING COMPREHENSIVE DATA VISUALIZATIONS")
        print("=" * 46)

        save_demographic_plots(final_df, DATA_DIR)
        save_behavioral_plots(final_df, DATA_DIR)
        save_socioeconomic_plots(final_df, DATA_DIR)
        save_institutional_plots(final_df, DATA_DIR)
        save_mindset_plots(final_df, DATA_DIR)

    except Exception as e:
        print(f"Execution Error: {e}")
