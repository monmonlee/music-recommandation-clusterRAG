"""
階層式 Clustering v3.0 - 改進版
- 更激進的 K 值搜尋
- 使用 Elbow Method
- 差異化策略
- 允許不分群
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score,
    calinski_harabasz_score
)
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 核心函數
# ============================================================

def find_elbow_point(k_range, inertias):
    """使用距離法找 Elbow Point"""
    if len(inertias) < 3:
        return k_range[0]
    
    # 正規化
    inertias_norm = (np.array(inertias) - min(inertias)) / (max(inertias) - min(inertias) + 1e-10)
    k_norm = (np.array(list(k_range)) - min(k_range)) / (max(k_range) - min(k_range) + 1e-10)
    
    # 計算到「理想線」的距離（從起點到終點的直線）
    distances = []
    for i in range(len(k_range)):
        # 點到線的距離公式
        x0, y0 = k_norm[i], inertias_norm[i]
        x1, y1 = k_norm[0], inertias_norm[0]
        x2, y2 = k_norm[-1], inertias_norm[-1]
        
        # 計算垂直距離
        d = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2 + 1e-10)
        distances.append(d)
    
    # 找最大距離（最大曲率）
    elbow_idx = np.argmax(distances)
    return k_range[elbow_idx]


def should_cluster(X, min_silhouette=0.13):
    """決定是否需要 clustering"""
    if X.shape[0] < 100:
        return False, 0.0
    
    # 測試 K=2 的品質
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=5)
    labels = kmeans.fit_predict(X)
    silhouette = silhouette_score(X, labels)
    
    return silhouette >= min_silhouette, silhouette


def find_optimal_k_v3(X, genre_name, n_songs, verbose=False):
    """
    v3: 改進版 K 值搜尋
    """
    
    # 檢查是否該分群
    should_do_clustering, test_sil = should_cluster(X)
    if not should_do_clustering:
        return 1, {'reason': 'low_quality', 'test_silhouette': test_sil}
    
    # 🎯 根據 Genre 類型決定策略
    electronic_genres = [
        'edm', 'house', 'techno', 'trance', 'dubstep', 
        'drum-and-bass', 'breakbeat', 'electro', 'electronic',
        'detroit-techno', 'minimal-techno', 'progressive-house',
        'chicago-house', 'deep-house'
    ]
    
    if genre_name in electronic_genres:
        min_k, max_k = 3, min(10, n_songs // 50)
        primary_metric = 'calinski'
    elif genre_name in ['classical', 'jazz', 'opera', 'piano', 'new-age']:
        min_k, max_k = 2, min(6, n_songs // 100)
        primary_metric = 'silhouette'
    else:
        min_k, max_k = 2, min(8, n_songs // 60)
        primary_metric = 'elbow'
    
    # 確保搜尋範圍合理
    max_k = max(max_k, 4)
    k_range = range(min_k, max_k + 1)
    
    # 計算各指標
    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_scores = []
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(X)
        
        silhouette_scores.append(silhouette_score(X, labels))
        davies_bouldin_scores.append(davies_bouldin_score(X, labels))
        calinski_scores.append(calinski_harabasz_score(X, labels))
        inertias.append(kmeans.inertia_)
    
    # 🔥 選擇最佳 K
    if primary_metric == 'elbow':
        best_k = find_elbow_point(k_range, inertias)
        selection_reason = 'elbow_method'
    elif primary_metric == 'silhouette':
        best_k = k_range[np.argmax(silhouette_scores)]
        selection_reason = 'max_silhouette'
    elif primary_metric == 'calinski':
        best_k = k_range[np.argmax(calinski_scores)]
        selection_reason = 'max_calinski'
    else:
        best_k = find_elbow_point(k_range, inertias)
        selection_reason = 'default_elbow'
    
    scores = {
        'k_range': list(k_range),
        'silhouette': silhouette_scores,
        'davies_bouldin': davies_bouldin_scores,
        'calinski': calinski_scores,
        'inertia': inertias,
        'best_k': best_k,
        'selection_reason': selection_reason,
        'primary_metric': primary_metric
    }
    
    if verbose:
        print(f"      K 搜尋範圍: {min_k}-{max_k}, 選擇 K={best_k} ({selection_reason})")
        for i, k in enumerate(k_range):
            marker = " ←" if k == best_k else ""
            print(f"        K={k}: Sil={silhouette_scores[i]:.3f}, "
                  f"CH={calinski_scores[i]:.0f}, Inertia={inertias[i]:.0f}{marker}")
    
    return best_k, scores


# ============================================================
# 主流程
# ============================================================

print("=" * 70)
print("🎵 階層式音樂分類系統 v3.0")
print("=" * 70)

df = pd.read_csv('dataset.csv')


audio_features = [
    'danceability', 'energy', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'loudness'
]
df_clean = df.dropna(subset=audio_features).copy()


genre_clusters = {}
clustering_summary = []

print("\n開始處理...")
print("=" * 70)

for idx, genre in enumerate(df_clean['track_genre'].unique(), 1):
    genre_data = df_clean[df_clean['track_genre'] == genre].copy()
    n_songs = len(genre_data)
    
    if n_songs < 50:
        genre_data['sub_cluster'] = 0
        genre_clusters[genre] = {
            'data': genre_data,
            'n_clusters': 1,
            'reason': 'too_few_songs'
        }
        print(f"[{idx:3d}/114] {genre:25s}: {n_songs:5,} 首 → 跳過（太少）")
        continue
    
    try:
        # 標準化
        X = genre_data[audio_features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 🔥 尋找最佳 K
        best_k, k_scores = find_optimal_k_v3(
            X_scaled, 
            genre, 
            n_songs,
            verbose=False  # 改成 True 可看詳細過程
        )
        
        if best_k == 1:
            # 不分群
            genre_data['sub_cluster'] = 0
            genre_clusters[genre] = {
                'data': genre_data,
                'n_clusters': 1,
                'reason': k_scores.get('reason', 'low_quality'),
                'test_silhouette': k_scores.get('test_silhouette', 0)
            }
            print(f"[{idx:3d}/114] {genre:25s}: {n_songs:5,} 首 → K=1 (品質太低，不分群)")
        else:
            # 正常分群
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20, max_iter=500)
            genre_data['sub_cluster'] = kmeans.fit_predict(X_scaled)
            
            final_sil = silhouette_score(X_scaled, genre_data['sub_cluster'])
            final_db = davies_bouldin_score(X_scaled, genre_data['sub_cluster'])
            
            genre_clusters[genre] = {
                'data': genre_data,
                'n_clusters': best_k,
                'model': kmeans,
                'scaler': scaler,
                'silhouette': final_sil,
                'davies_bouldin': final_db,
                'k_scores': k_scores
            }
            
            clustering_summary.append({
                'genre': genre,
                'n_songs': n_songs,
                'n_clusters': best_k,
                'silhouette': final_sil,
                'davies_bouldin': final_db,
                'selection_reason': k_scores['selection_reason']
            })
            
            print(f"[{idx:3d}/114] {genre:25s}: {n_songs:5,} 首 → K={best_k} "
                  f"(Sil: {final_sil:.3f}, {k_scores['selection_reason']})")
    
    except Exception as e:
        genre_data['sub_cluster'] = 0
        genre_clusters[genre] = {
            'data': genre_data,
            'n_clusters': 1,
            'reason': f'error: {str(e)[:30]}'
        }
        print(f"[{idx:3d}/114] {genre:25s}: 錯誤")

# 統計摘要
print("\n" + "=" * 70)
print("統計摘要")
print("=" * 70)

if clustering_summary:
    summary_df = pd.DataFrame(clustering_summary)
    
    print(f"\nK 值分布:")
    k_dist = summary_df['n_clusters'].value_counts().sort_index()
    for k, count in k_dist.items():
        print(f"  K={k}: {count:3d} 個 Genre ({count/len(summary_df)*100:5.1f}%)")
    
    print(f"\n品質統計:")
    print(f"  平均 Silhouette: {summary_df['silhouette'].mean():.4f}")
    print(f"  平均 Davies-Bouldin: {summary_df['davies_bouldin'].mean():.4f}")
    
    # 儲存
    df_final = pd.concat([info['data'] for info in genre_clusters.values()], ignore_index=True)
    df_final.to_parquet('data/tracks_with_hierarchical_clusters_v3.parquet', index=False)
    joblib.dump(genre_clusters, 'models/genre_clusters_v3.pkl')
    summary_df.to_csv('results/hierarchical_clustering_summary_v3.csv', index=False)
    
    print("\n✅ 完成！")
