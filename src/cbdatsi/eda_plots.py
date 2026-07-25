# src/cbdatsi/eda_plots.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_demographics(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    def plot_bar(ax, series, title, order=None, color='#2F5597'):
        counts = series.value_counts().reindex(order).fillna(0) if order else series.value_counts()
        ax.bar(counts.index.astype(str), counts.values, color=color, edgecolor='black', alpha=0.85)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Students')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.tick_params(axis='x', rotation=15)

    plot_bar(axes[0], df['Year_Label'], 'Distribution by Year Level', ['First-year', 'Second-year', 'Third-year', 'Fourth-year', 'Graduated'], '#2F5597')
    plot_bar(axes[1], df['Gender_Label'], 'Distribution by Gender', None, '#8FAADC')
    plot_bar(axes[2], df['GPA_Label'], 'Baseline GPA Distribution', ['Poor', 'Average', 'Fair', 'Good', 'Excellent'], '#4472C4')
    plt.tight_layout()
    plt.show()

def plot_behavioral_boxplots(df: pd.DataFrame):
    """Replaces EDA 2 bar charts with Boxplots to show Median/IQR"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    gpa_order = ['Poor', 'Average', 'Fair', 'Good', 'Excellent']
    
    sns.boxplot(x='GPA_Label', y='Time_Studying', hue='GPA_Label', data=df, 
                order=gpa_order, palette='Greens', ax=axes[0], legend=False)
    axes[0].set_title('Study Time Distribution Across GPA Brackets', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Study Time Level (1-5 Scale)')
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    sns.boxplot(x='GPA_Label', y='Time_SocicalMedia', hue='GPA_Label', data=df, 
                order=gpa_order, palette='Oranges', ax=axes[1], legend=False)
    axes[1].set_title('Social Media Time Distribution Across GPA Brackets', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Social Media Level (1-5 Scale)')
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def plot_socioeconomic_conditional(df: pd.DataFrame):
    """Shows poor/not-poor percentage WITHIN each GPA level"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    gpa_order = ['Poor', 'Average', 'Fair', 'Good', 'Excellent']

    ct_wealth = pd.crosstab(df['GPA_Label'], df['Poor_Stu_Label'], normalize='index').reindex(gpa_order) * 100
    ct_wealth.plot(kind='bar', ax=axes[0], color=['#8C2D19', '#1F4E79'], edgecolor='black', width=0.7)
    axes[0].set_title('Household Wealth Breakdown Per GPA Level', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Percentage within GPA Tier (%)')
    axes[0].set_xticklabels(gpa_order, rotation=0)

    ct_policy = pd.crosstab(df['GPA_Label'], df['Policy_Stu_Label'], normalize='index').reindex(gpa_order) * 100
    ct_policy.plot(kind='bar', ax=axes[1], color=['#27408B', '#CD8500'], edgecolor='black', width=0.7)
    axes[1].set_title('Policy Support Breakdown Per GPA Level', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Percentage within GPA Tier (%)')
    axes[1].set_xticklabels(gpa_order, rotation=0)
    plt.tight_layout()
    plt.show()

def plot_institutional_heatmap(df: pd.DataFrame):
    inst_cols = ['SupportOf_Uni', 'SupportOf_Lec', 'Facilitie_Uni', 'Quality_Lecturer', 'TrainingCurriculum', 'GPA']
    gpa_corr = df[inst_cols].corr(method='spearman')[['GPA']].drop('GPA').sort_values(by='GPA', ascending=False)
    plt.figure(figsize=(6, 5))
    sns.heatmap(gpa_corr, annot=True, cmap='coolwarm', fmt=".3f", vmin=-0.2, vmax=0.2, linewidths=1.5, linecolor='black')
    plt.title('Institutional Factors vs. Student GPA', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.show()

def plot_mindset_heatmaps(df: pd.DataFrame):
    ct_study = pd.crosstab(df['GPA'], df['Study_Methods'], normalize='columns')
    ct_comp = pd.crosstab(df['GPA'], df['Competitive_Class'], normalize='columns')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(ct_study, annot=True, fmt=".0%", cmap="Blues", ax=axes[0])
    axes[0].set_title('GPA Distribution by Study Methods (Internal)', fontweight='bold')
    axes[0].invert_yaxis() 
    sns.heatmap(ct_comp, annot=True, fmt=".0%", cmap="Oranges", ax=axes[1])
    axes[1].set_title('GPA Distribution by Class Competition (External)', fontweight='bold')
    axes[1].invert_yaxis()
    plt.tight_layout()
    plt.show()