import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def preprocess_unsupervised(df, feature_cols=None):
    """Preprocess data for unsupervised learning."""
    df = df.copy()
    
    if feature_cols is not None:
        if len(feature_cols) == 0:
            raise ValueError("No feature columns selected. Please select at least one feature for clustering.")
        df = df[feature_cols]
    
    # Identify numerical and categorical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(exclude=['int64', 'float64']).columns
    
    # Impute missing values
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy='mean')
        df[num_cols] = num_imputer.fit_transform(df[num_cols])
        
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        
    # Scale features using RobustScaler (consistent with supervised pipeline)
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    return X_scaled

def auto_kmeans_clustering(X_scaled, max_k=10):
    """Baseline: Find the best K for KMeans using Silhouette score."""
    best_k = 2
    best_score = -1
    best_model = None
    best_labels = None
    
    max_k = min(max_k, len(X_scaled) - 1)
    
    scores = []
    inertias = []
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=500)
        labels = kmeans.fit_predict(X_scaled)
        
        inertias.append((k, kmeans.inertia_))
        
        if len(set(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
        else:
            score = -1
            
        scores.append((k, score))
        
        if score > best_score:
            best_score = score
            best_k = k
            best_model = kmeans
            best_labels = labels
            
    # Apply PCA for visualization
    n_components = min(3, X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    
    if X_scaled.shape[1] < 2:
        pca_result = pd.DataFrame({'PCA1': X_scaled.iloc[:, 0], 'PCA2': [0]*len(X_scaled)})
        pca_df = pca_result
        explained_variance = [1.0]
    else:
        pca_result = pca.fit_transform(X_scaled)
        if n_components >= 3:
            pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2', 'PCA3'])
        else:
            pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
        explained_variance = pca.explained_variance_ratio_.tolist()
        
    pca_df['Cluster'] = best_labels
    pca_df['Cluster'] = pca_df['Cluster'].astype(str)
    
    scores_df = pd.DataFrame(scores, columns=['K', 'Silhouette Score'])
    inertias_df = pd.DataFrame(inertias, columns=['K', 'Inertia'])
    
    return best_k, best_score, scores_df, inertias_df, pca_df, best_model, explained_variance


def auto_tune_clustering(X_scaled, max_k=10):
    """
    Auto-tuned clustering: Tests multiple clustering algorithms with various hyperparameters.
    Returns the best algorithm, its config, comparison results, and PCA visualization data.
    """
    max_k = min(max_k, len(X_scaled) - 1)
    all_results = []
    best_overall_score = -1
    best_overall_labels = None
    best_overall_name = ""
    best_overall_config = {}
    
    # --- 1. KMeans with extended hyperparameter sweep ---
    kmeans_inertias = []
    kmeans_scores = []
    for k in range(2, max_k + 1):
        for n_init in [10, 20]:
            for max_iter in [300, 500]:
                try:
                    model = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=42)
                    labels = model.fit_predict(X_scaled)
                    if len(set(labels)) > 1:
                        score = silhouette_score(X_scaled, labels)
                    else:
                        score = -1
                    
                    all_results.append({
                        'Algorithm': 'KMeans',
                        'Config': f'K={k}, n_init={n_init}, max_iter={max_iter}',
                        'Clusters': k,
                        'Silhouette Score': round(score, 4),
                        '_labels': labels,
                        '_params': {'n_clusters': k, 'n_init': n_init, 'max_iter': max_iter}
                    })
                    kmeans_inertias.append((k, model.inertia_))
                    kmeans_scores.append((k, score))
                    
                    if score > best_overall_score:
                        best_overall_score = score
                        best_overall_labels = labels
                        best_overall_name = 'KMeans'
                        best_overall_config = {'n_clusters': k, 'n_init': n_init, 'max_iter': max_iter}
                except Exception:
                    pass
    
    # --- 2. MiniBatchKMeans (faster, good for large datasets) ---
    for k in range(2, max_k + 1):
        for batch_size in [100, 256, 512]:
            try:
                model = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=42, n_init=10)
                labels = model.fit_predict(X_scaled)
                if len(set(labels)) > 1:
                    score = silhouette_score(X_scaled, labels)
                else:
                    score = -1
                
                all_results.append({
                    'Algorithm': 'MiniBatchKMeans',
                    'Config': f'K={k}, batch={batch_size}',
                    'Clusters': k,
                    'Silhouette Score': round(score, 4),
                    '_labels': labels,
                    '_params': {'n_clusters': k, 'batch_size': batch_size}
                })
                
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall_labels = labels
                    best_overall_name = 'MiniBatchKMeans'
                    best_overall_config = {'n_clusters': k, 'batch_size': batch_size}
            except Exception:
                pass
    
    # --- 3. Agglomerative Clustering ---
    for k in range(2, max_k + 1):
        for linkage in ['ward', 'complete', 'average']:
            try:
                model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                labels = model.fit_predict(X_scaled)
                if len(set(labels)) > 1:
                    score = silhouette_score(X_scaled, labels)
                else:
                    score = -1
                
                all_results.append({
                    'Algorithm': 'Agglomerative',
                    'Config': f'K={k}, linkage={linkage}',
                    'Clusters': k,
                    'Silhouette Score': round(score, 4),
                    '_labels': labels,
                    '_params': {'n_clusters': k, 'linkage': linkage}
                })
                
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall_labels = labels
                    best_overall_name = 'Agglomerative'
                    best_overall_config = {'n_clusters': k, 'linkage': linkage}
            except Exception:
                pass
    
    # --- 4. Gaussian Mixture Model ---
    for k in range(2, max_k + 1):
        for cov_type in ['full', 'tied', 'diag', 'spherical']:
            try:
                model = GaussianMixture(n_components=k, covariance_type=cov_type, random_state=42)
                model.fit(X_scaled)
                labels = model.predict(X_scaled)
                if len(set(labels)) > 1:
                    score = silhouette_score(X_scaled, labels)
                else:
                    score = -1
                
                all_results.append({
                    'Algorithm': 'Gaussian Mixture',
                    'Config': f'K={k}, cov={cov_type}',
                    'Clusters': k,
                    'Silhouette Score': round(score, 4),
                    '_labels': labels,
                    '_params': {'n_components': k, 'covariance_type': cov_type}
                })
                
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall_labels = labels
                    best_overall_name = 'Gaussian Mixture'
                    best_overall_config = {'n_components': k, 'covariance_type': cov_type}
            except Exception:
                pass
    
    # --- 5. DBSCAN (density-based, auto-determines clusters) ---
    for eps in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        for min_samples in [3, 5, 10]:
            try:
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(X_scaled)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters >= 2:
                    # Filter out noise points for silhouette
                    mask = labels != -1
                    if mask.sum() > n_clusters:
                        score = silhouette_score(X_scaled[mask], labels[mask])
                    else:
                        score = -1
                else:
                    score = -1
                
                all_results.append({
                    'Algorithm': 'DBSCAN',
                    'Config': f'eps={eps}, min_samples={min_samples}',
                    'Clusters': n_clusters,
                    'Silhouette Score': round(score, 4),
                    '_labels': labels,
                    '_params': {'eps': eps, 'min_samples': min_samples}
                })
                
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall_labels = labels
                    best_overall_name = 'DBSCAN'
                    best_overall_config = {'eps': eps, 'min_samples': min_samples, 'clusters_found': n_clusters}
            except Exception:
                pass
    
    # --- Build comparison results (best per algorithm) ---
    results_for_display = []
    seen_algos = {}
    for r in all_results:
        algo = r['Algorithm']
        if algo not in seen_algos or r['Silhouette Score'] > seen_algos[algo]['Silhouette Score']:
            seen_algos[algo] = r
    
    for algo, r in seen_algos.items():
        results_for_display.append({
            'Algorithm': r['Algorithm'],
            'Best Config': r['Config'],
            'Clusters': r['Clusters'],
            'Silhouette Score': r['Silhouette Score']
        })
    
    comparison_df = pd.DataFrame(results_for_display).sort_values('Silhouette Score', ascending=False).reset_index(drop=True)
    
    # --- KMeans inertia/silhouette for Elbow chart ---
    # Deduplicate: keep best score per K
    k_scores = {}
    for k, s in kmeans_scores:
        if k not in k_scores or s > k_scores[k]:
            k_scores[k] = s
    scores_df = pd.DataFrame(sorted(k_scores.items()), columns=['K', 'Silhouette Score'])
    
    k_inertias = {}
    for k, i in kmeans_inertias:
        if k not in k_inertias or i < k_inertias[k]:
            k_inertias[k] = i
    inertias_df = pd.DataFrame(sorted(k_inertias.items()), columns=['K', 'Inertia'])
    
    # --- PCA visualization with best labels ---
    n_components = min(3, X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    
    if X_scaled.shape[1] < 2:
        pca_df = pd.DataFrame({'PCA1': X_scaled.iloc[:, 0], 'PCA2': [0]*len(X_scaled)})
        explained_variance = [1.0]
    else:
        pca_result = pca.fit_transform(X_scaled)
        if n_components >= 3:
            pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2', 'PCA3'])
        else:
            pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
        explained_variance = pca.explained_variance_ratio_.tolist()
    
    pca_df['Cluster'] = best_overall_labels
    pca_df['Cluster'] = pca_df['Cluster'].astype(str)
    
    best_k = best_overall_config.get('n_clusters', best_overall_config.get('n_components', 
              best_overall_config.get('clusters_found', len(set(best_overall_labels)) - (1 if -1 in best_overall_labels else 0))))
    
    return (best_k, best_overall_score, best_overall_name, best_overall_config,
            comparison_df, scores_df, inertias_df, pca_df, explained_variance)
