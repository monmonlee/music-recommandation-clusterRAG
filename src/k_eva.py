"""
階層式 Clustering 評估腳本
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import time

# 設定繪圖風格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 70)
print("📊 階層式 Clustering 評估")
print("=" * 70)

# ============================================================
# 1. 載入資料
# ============================================================
print("\n[載入資料]")
df = pd.read_parquet('data/tracks_with_hierarchical_clusters_v3.parquet')
genre_clusters = joblib.load('models/genre_clusters.pkl')

print(f"✅ 資料：{len(df):,} 首歌")
print(f"✅ Genre 數量：{df['track_genre'].nunique()}")
print(f"✅ 總次分類數：{len(df.groupby(['track_genre', 'sub_cluster']))}")

audio_features = [
    'danceability', 'energy', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'loudness'
]

# ============================================================
# 2. 對比：全局 vs 階層式
# ============================================================
print("\n" + "=" * 70)
print("[評估 1] 全局 vs 階層式 Clustering 對比")
print("=" * 70)

# 全局 K-means 結果（從你之前的實驗）
global_metrics = {
    'method': 'Global K-means (K=100)',
    'silhouette': 0.124,
    'davies_bouldin': 1.552,
    'avg_cluster_size': 1140,
    'genre_purity': 0.152
}

# 階層式結果
successful_genres = [
    info for info in genre_clusters.values() 
    if info['reason'] == 'success'
]

hierarchical_metrics = {
    'method': 'Hierarchical K-means',
    'silhouette': np.mean([info['silhouette'] for info in successful_genres]),
    'davies_bouldin': np.mean([info['davies_bouldin'] for info in successful_genres]),
    'avg_cluster_size': df.groupby(['track_genre', 'sub_cluster']).size().mean(),
    'genre_purity': 1.0  # 100% 因為在同 genre 內
}

comparison_df = pd.DataFrame([global_metrics, hierarchical_metrics])
print("\n" + comparison_df.to_string(index=False))

# 計算改善程度
sil_improvement = (hierarchical_metrics['silhouette'] - global_metrics['silhouette']) / global_metrics['silhouette'] * 100
db_improvement = (global_metrics['davies_bouldin'] - hierarchical_metrics['davies_bouldin']) / global_metrics['davies_bouldin'] * 100

print(f"\n📈 改善程度:")
print(f"   Silhouette Score: +{sil_improvement:.1f}%")
print(f"   Davies-Bouldin:   -{db_improvement:.1f}% (越低越好)")
print(f"   Genre 純度:       +{(1.0 - global_metrics['genre_purity']) * 100:.1f}% → 100%")

# ============================================================
# 3. 視覺化：PCA 對比
# ============================================================
print("\n" + "=" * 70)
print("[評估 2] 視覺化主要 Genre 的次分類")
print("=" * 70)

def visualize_genre_subclusters(genre, save=True):
    """視覺化某個 genre 的次分類"""
    if genre not in genre_clusters:
        print(f"⚠️  {genre} 不存在")
        return
    
    cluster_info = genre_clusters[genre]
    if cluster_info['n_clusters'] == 1:
        print(f"⚠️  {genre} 只有 1 個分類，跳過視覺化")
        return
    
    data = cluster_info['data']
    X = data[audio_features].values
    scaler = cluster_info['scaler']
    X_scaled = scaler.transform(X)
    
    # PCA 降維
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # 繪圖
    plt.figure(figsize=(12, 8))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    for i in range(cluster_info['n_clusters']):
        mask = data['sub_cluster'].values == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=colors[i % len(colors)], 
                   label=f'次分類 {i} ({mask.sum()} 首)',
                   alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    plt.title(f'{genre.upper()} 的次分類視覺化 (K={cluster_info["n_clusters"]})', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save:
        plt.savefig(f'results/pca_{genre}.png', dpi=300, bbox_inches='tight')
        print(f"✅ {genre}: 已儲存 results/pca_{genre}.png")
    
    plt.close()

# 視覺化主要 Genre
major_genres = ['pop', 'rock', 'edm', 'hip-hop', 'jazz', 'classical']
for genre in major_genres:
    visualize_genre_subclusters(genre)

# ============================================================
# 4. 檢索效能測試
# ============================================================
print("\n" + "=" * 70)
print("[評估 3] 檢索效能測試")
print("=" * 70)

def test_retrieval(target_song_name, methods=['global', 'genre', 'hierarchical'], top_k=10):
    """測試不同檢索方法"""
    # 找到目標歌曲
    target_candidates = df[df['track_name'].str.contains(target_song_name, case=False, na=False)]
    
    if len(target_candidates) == 0:
        print(f"❌ 找不到歌曲：{target_song_name}")
        return None
    
    target = target_candidates.iloc[0]
    target_features = target[audio_features].values
    
    results = {}
    
    for method in methods:
        start_time = time.time()
        
        if method == 'global':
            # 全局檢索
            candidates = df
        elif method == 'genre':
            # Genre 過濾
            candidates = df[df['track_genre'] == target['track_genre']]
        elif method == 'hierarchical':
            # Genre + 次分類過濾
            candidates = df[
                (df['track_genre'] == target['track_genre']) &
                (df['sub_cluster'] == target['sub_cluster'])
            ]
        else:
            continue
        
        # 計算距離
        distances = np.linalg.norm(
            candidates[audio_features].values - target_features,
            axis=1
        )
        
        # 排除自己
        non_self_mask = candidates.index != target.name
        distances_filtered = distances[non_self_mask]
        candidates_filtered = candidates[non_self_mask]
        
        # 取前 K 首
        if len(distances_filtered) >= top_k:
            top_k_idx = distances_filtered.argsort()[:top_k]
            top_k_songs = candidates_filtered.iloc[top_k_idx]
            avg_distance = distances_filtered[top_k_idx].mean()
        else:
            top_k_songs = candidates_filtered
            avg_distance = distances_filtered.mean() if len(distances_filtered) > 0 else np.inf
        
        time_cost = time.time() - start_time
        
        results[method] = {
            'top_k': top_k_songs,
            'avg_distance': avg_distance,
            'time': time_cost,
            'pool_size': len(candidates)
        }
    
    return target, results

# 測試歌曲
test_songs = [
    'Shape of You',
    'Bohemian Rhapsody',
    'Blinding Lights',
    'Smells Like Teen Spirit',
    'Billie Jean'
]

print("\n測試歌曲檢索效能：\n")

retrieval_stats = []

for song_name in test_songs:
    print(f"🎵 目標歌曲: {song_name}")
    result = test_retrieval(song_name)
    
    if result is None:
        continue
    
    target, methods_results = result
    print(f"   Genre: {target['track_genre']}")
    print(f"   次分類: {target['sub_cluster']}")
    print()
    
    for method, res in methods_results.items():
        retrieval_stats.append({
            'song': song_name,
            'method': method,
            'avg_distance': res['avg_distance'],
            'time_ms': res['time'] * 1000,
            'pool_size': res['pool_size']
        })
        
        print(f"   【{method:12s}】 "
              f"Pool: {res['pool_size']:6,} | "
              f"Time: {res['time']*1000:6.2f}ms | "
              f"Avg Dist: {res['avg_distance']:.4f}")
    
    print()

# 統計摘要
if retrieval_stats:
    stats_df = pd.DataFrame(retrieval_stats)
    
    print("\n📊 檢索效能統計摘要:")
    print("=" * 70)
    summary = stats_df.groupby('method').agg({
        'avg_distance': 'mean',
        'time_ms': 'mean',
        'pool_size': 'mean'
    }).round(2)
    print(summary)
    
    # 計算加速比
    global_time = summary.loc['global', 'time_ms']
    hierarchical_time = summary.loc['hierarchical', 'time_ms']
    speedup = global_time / hierarchical_time
    
    print(f"\n⚡ 加速比: {speedup:.2f}x")
    print(f"   (階層式比全局快 {speedup:.2f} 倍)")
    
    # 儲存結果
    stats_df.to_csv('results/retrieval_performance.csv', index=False)
    print("\n✅ 結果已儲存：results/retrieval_performance.csv")

# ============================================================
# 5. 生成推薦對比（供人工評估）
# ============================================================
print("\n" + "=" * 70)
print("[評估 4] 生成推薦對比範例")
print("=" * 70)

def generate_comparison_sample(song_name, output_file='results/recommendation_comparison.txt'):
    """生成推薦對比範例"""
    result = test_retrieval(song_name, top_k=5)
    
    if result is None:
        return
    
    target, methods_results = result
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"目標歌曲: {target['track_name']} - {target['artists']}\n")
        f.write(f"Genre: {target['track_genre']} | 次分類: {target['sub_cluster']}\n")
        f.write("=" * 70 + "\n\n")
        
        for method, res in methods_results.items():
            f.write(f"【{method.upper()}】 候選池大小: {res['pool_size']:,}\n")
            f.write("-" * 50 + "\n")
            
            for i, (_, song) in enumerate(res['top_k'].iterrows(), 1):
                f.write(f"{i}. {song['track_name']:40s} - {song['artists']}\n")
            
            f.write("\n")
        
        f.write("請評分 (1-5分):\n")
        f.write("  Global:       ___\n")
        f.write("  Genre:        ___\n")
        f.write("  Hierarchical: ___\n")
        f.write("\n\n")

# 清空舊檔案
with open('results/recommendation_comparison.txt', 'w', encoding='utf-8') as f:
    f.write("音樂檢索推薦對比評估\n")
    f.write(f"生成時間: {pd.Timestamp.now()}\n\n")

# 生成對比範例
for song in test_songs[:3]:  # 只生成前 3 首
    generate_comparison_sample(song)

print("✅ 推薦對比已儲存：results/recommendation_comparison.txt")
print("   請人工評估推薦品質並填寫評分")

# ============================================================
# 6. 完成
# ============================================================
print("\n" + "=" * 70)
print("🎉 評估完成！")
print("=" * 70)
print("\n產出檔案：")
print("  📊 results/pca_*.png (各 Genre 的視覺化)")
print("  📈 results/retrieval_performance.csv")
print("  📝 results/recommendation_comparison.txt")