import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_lr_tuning_results(grid_lr):
    """Generates the hyperparameter tuning line plot for Ordinal Logistic Regression."""
    results_lr = pd.DataFrame(grid_lr.cv_results_)
    results_lr['C'] = results_lr['param_classifier__C'].astype(float)
    results_lr['Penalty'] = results_lr['param_classifier__penalty'].astype(str).str.upper()
    results_lr['Mean F1'] = results_lr['mean_test_score']

    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=results_lr, 
        x='C', 
        y='Mean F1', 
        hue='Penalty', 
        marker='o', 
        linewidth=2.5, 
        palette=['#e74c3c', '#3498db']
    )
    plt.xscale('log')

    c_ticks = sorted(results_lr['C'].unique())
    c_labels = [f"{c:g}" for c in c_ticks]
    plt.xticks(c_ticks, c_labels)

    plt.title('Ordinal Logistic Regression: Hyperparameter Tuning', fontsize=13, fontweight='bold')
    plt.xlabel('C (Inverse Regularization Strength)')
    plt.ylabel('Macro F1-Score (Cross-Validation)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_mlp_tuning_heatmap(results_df, architecture_keys):
    """Generates the heatmap for MLP architecture vs activation tuning."""
    pivot_mlp = results_df.pivot(
        index='Architecture', 
        columns='Activation', 
        values='Mean F1'
    ).reindex(architecture_keys)

    plt.figure(figsize=(9, 7))
    sns.heatmap(pivot_mlp, annot=True, cmap='viridis', fmt=".4f", linewidths=1, linecolor='black')
    plt.title('MLP Macro F1: Architectures vs Activations', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_rf_tuning_heatmap(grid_rf):
    """Generates the heatmap for Random Forest depth vs tree/split configuration tuning."""
    results_rf = pd.DataFrame(grid_rf.cv_results_)
    results_rf['max_depth'] = results_rf['param_classifier__max_depth'].fillna('None')
    results_rf['Config'] = results_rf.apply(
        lambda r: f"Trees: {r['param_classifier__n_estimators']} | Split: {r['param_classifier__min_samples_split']}", 
        axis=1
    )

    pivot_rf = results_rf.pivot(
        index='max_depth', 
        columns='Config', 
        values='mean_test_score'
    ).reindex([10, 20, 'None'])

    plt.figure(figsize=(9, 5))
    sns.heatmap(pivot_rf, annot=True, cmap='YlGnBu', fmt=".4f", linewidths=1, linecolor='black')
    plt.title('Random Forest Tuning: Depth vs Split & Trees', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_model_metrics_comparison(metrics_list, model_names):
    """Plots side-by-side bar charts for Macro F1, MAE, and QWK across models."""
    f1s = [m['Macro_F1'] for m in metrics_list]
    maes = [m['MAE'] for m in metrics_list]
    qwks = [m['QWK'] for m in metrics_list]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Macro F1
    sns.barplot(x=model_names, y=f1s, ax=axes[0], palette='viridis', edgecolor='black')
    axes[0].set_title('Test Set: Macro F1-Score (Higher is Better)', fontweight='bold')
    axes[0].set_ylim(0, max(f1s) * 1.25)
    for i, v in enumerate(f1s): 
        axes[0].text(i, v + 0.005, f"{v:.4f}", ha='center', fontweight='bold')

    # MAE
    sns.barplot(x=model_names, y=maes, ax=axes[1], palette='magma', edgecolor='black')
    axes[1].set_title('Test Set: MAE (Lower is Better)', fontweight='bold')
    axes[1].set_ylim(0, max(maes) * 1.20)
    for i, v in enumerate(maes): 
        axes[1].text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')

    # QWK
    sns.barplot(x=model_names, y=qwks, ax=axes[2], palette='mako', edgecolor='black')
    axes[2].set_title('Test Set: QWK (Higher is Better)', fontweight='bold')
    axes[2].set_ylim(min(0, min(qwks)), max(qwks) * 1.25 if max(qwks) > 0 else 0.5)
    for i, v in enumerate(qwks): 
        axes[2].text(i, v + 0.005, f"{v:.4f}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(metrics_list, model_names, class_labels=[1, 2, 3, 4, 5]):
    """Plots side-by-side confusion matrix heatmaps for all evaluated models."""
    cms = [m['Confusion_Matrix'] for m in metrics_list]

    fig_cm, axes_cm = plt.subplots(1, len(model_names), figsize=(18, 4))

    for i, (cm, model_name) in enumerate(zip(cms, model_names)):
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            cbar=False,
            annot_kws={'size': 10, 'weight': 'bold'},
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=axes_cm[i]
        )
        axes_cm[i].set_title(f'{model_name}', fontweight='bold', fontsize=12)
        axes_cm[i].set_xlabel('Predicted GPA Bracket', fontweight='bold')
        if i == 0:
            axes_cm[i].set_ylabel('Actual GPA Bracket', fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_feature_weights_heatmap(weights_df):
    """
    Generates a concise matrix heatmap comparing normalized feature weights across all models.
    """
    model_cols = [col for col in weights_df.columns if col != 'Overall Mean Weight']
    sorted_df = weights_df[model_cols].loc[weights_df[model_cols].mean(axis=1).sort_values(ascending=False).index]

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        sorted_df, 
        annot=True, 
        fmt=".3f", 
        cmap="YlGnBu", 
        linewidths=1, 
        linecolor="black",
        cbar_kws={'label': 'Normalized Feature Weight'}
    )
    plt.title('Feature Weight Mapping Summary Matrix Across Models', fontsize=13, fontweight='bold')
    plt.xlabel('Model Architecture', fontweight='bold', fontsize=11)
    plt.ylabel('Selected Features', fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_average_feature_weights(weights_df):
    """
    Generates a horizontal bar chart displaying the overall average feature weight across all evaluated models.
    """
    model_cols = [col for col in weights_df.columns if col != 'Overall Mean Weight']
    mean_weights = weights_df[model_cols].mean(axis=1).sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(mean_weights))
    
    bars = plt.barh(mean_weights.index, mean_weights.values, color=colors, edgecolor='black', alpha=0.85)
    
    # Value annotations on each bar
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.003, 
            bar.get_y() + bar.get_height() / 2, 
            f"{width:.4f}", 
            va='center', 
            ha='left', 
            fontweight='bold', 
            fontsize=10
        )

    plt.title('Overall Mean Feature Weight Mapping (Averaged Across Models)', fontsize=13, fontweight='bold')
    plt.xlabel('Mean Normalized Importance Weight', fontsize=11, fontweight='bold')
    plt.ylabel('Selected Feature Vectors', fontsize=11, fontweight='bold')
    plt.xlim(0, max(mean_weights.values) * 1.15)
    plt.grid(True, linestyle='--', alpha=0.5, axis='x')
    plt.tight_layout()
    plt.show()