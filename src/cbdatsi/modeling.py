# src/cbdatsi/modeling.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

def get_clustering_features() -> list:
    """Returns the selected internal and external features for profiling."""
    return [
        'Study_Methods', 'Time_Studying', 'Time_Friends', 'Time_SocicalMedia', 
        'Adapt_Learning_Uni', 'Policy_Stu', 'SupportOf_Uni', 'SupportOf_Lec', 
        'Facilitie_Uni', 'Quality_Lecturer', 'TrainingCurriculum'
    ]

def preprocess_clustering_data(df: pd.DataFrame) -> pd.DataFrame:
    """Isolates features and applies binary encoding to Policy_Stu."""
    features = get_clustering_features()
    X = df[features].copy()
    
    if 'Policy_Stu' in X.columns:
        X['Policy_Stu'] = X['Policy_Stu'].replace({2: 0})
        
    return X

def plot_elbow_curve(df: pd.DataFrame, max_k: int = 10):
    """Generates and displays the Elbow Method WCSS curve."""
    X = preprocess_clustering_data(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    wcss = []
    k_range = range(1, max_k + 1)
    for k in k_range:
        kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_test.fit(X_scaled)
        wcss.append(kmeans_test.inertia_)
        
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, wcss, marker='o', linestyle='--', color='#2F5597')
    plt.title('Elbow Method For Optimal k', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def run_kmeans_clustering(df: pd.DataFrame, n_clusters: int = 3) -> tuple:
    """Applies feature scaling, runs K-Means, and returns the dataframe with labels."""
    df_clustered = df.copy()
    
    X = preprocess_clustering_data(df_clustered)
    features = get_clustering_features()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_clustered['Cluster'] = kmeans.fit_predict(X_scaled)
    
    df_clustered['Policy_Stu'] = df_clustered['Policy_Stu'].replace({2: 0})
    cluster_summary = df_clustered.groupby('Cluster')[features + ['GPA']].mean()
    
    return df_clustered, cluster_summary

def plot_cluster_pca(df: pd.DataFrame):
    """Reduces 11D cluster data to 2D using PCA and plots the clusters."""
    X = preprocess_clustering_data(df)
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    
    plot_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
    plot_df['Cluster'] = df['Cluster'].astype(str)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette=['#1f77b4', '#ff7f0e', '#2ca02c'], 
                    data=plot_df, alpha=0.7)
    plt.title('2D PCA Visualization of Student Clusters', fontweight='bold')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_cluster_profiles(cluster_summary: pd.DataFrame):
    """Plots a grouped bar chart comparing feature means and GPA across clusters."""
    features_and_gpa = cluster_summary.T
    
    features_and_gpa.plot(kind='bar', figsize=(14, 6), colormap='viridis', edgecolor='black')
    plt.title('Feature Means and GPA by Student Cluster', fontsize=14, fontweight='bold')
    plt.ylabel('Mean Value')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def evaluate_clusters(df: pd.DataFrame):
    """Calculates and prints the Silhouette Score for the clustered data."""
    X = preprocess_clustering_data(df)
    X_scaled = StandardScaler().fit_transform(X)
    labels = df['Cluster']
    
    score = silhouette_score(X_scaled, labels)
    print(f"--- K-Means Model Evaluation ---")
    print(f"Silhouette Score: {score:.4f}")
    return score