# src/cbdatsi/modeling.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def get_clustering_features() -> list:
    """Returns the selected internal and external features for profiling."""
    return [
        'Study_Methods', 'Time_Studying', 'Time_Friends', 'Time_SocicalMedia', 
        'Adapt_Learning_Uni', 'Policy_Stu', 'SupportOf_Uni', 'SupportOf_Lec', 
        'Facilitie_Uni', 'Quality_Lecturer', 'TrainingCurriculum'
    ]

def plot_elbow_curve(df: pd.DataFrame, max_k: int = 10):
    """Generates and displays the Elbow Method WCSS curve."""
    features = get_clustering_features()
    X = df[features].copy()
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
    """Applies feature scaling, runs K-Means, and returns the dataframe with labels and cluster summaries."""
    df_clustered = df.copy()
    features = get_clustering_features()
    
    X = df_clustered[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_clustered['Cluster'] = kmeans.fit_predict(X_scaled)
    
    cluster_summary = df_clustered.groupby('Cluster')[features + ['GPA']].mean()
    return df_clustered, cluster_summary