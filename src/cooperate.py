

# ============================================================================
# 套件導入
# ============================================================================
import os
import json
import time
import uuid
import shutil
import datetime
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import anthropic
import gradio as gr
from router import QueryRouter, QueryType
import joblib
import pandas as pd


# ============================================================================
# 全域設定（已改為本機版本‼）
# ============================================================================

# 將這個改成你自己的本機資料夾位置（你放 dataset.csv、chroma_db 的地方）
# 建議你把這個資料夾命名為 `spotify_data/`
DRIVE_ROOT = "/Users/mangtinglee/Desktop/UT/data mining/期末"

DATA_PATH = os.path.join(DRIVE_ROOT, "dataset.csv")
DB_PATH = os.path.join(DRIVE_ROOT, "chroma_db")
LOG_PATH = os.path.join(DRIVE_ROOT, "query_log.txt")


# API 設定
_ = load_dotenv(find_dotenv())
ANTHROPIC_API_KEY = os.environ['ANTHROPIC_API_KEY']   # ← 改成你自己的 key
CLAUDE_MODEL = "claude-3-5-haiku-20241022"  # 建議改用最新模型

# 推薦數量設定
CANDIDATE_SIZE = 50
FINAL_RECOMMENDATION = 10

# ============================================================================
# 初始化全域物件
# ============================================================================
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

chroma_client = None
collections = {}
chat_sessions = {}


# ============================================================================
# 資料處理函數
# ============================================================================
def normalize_numeric(row):
    """
    將數值欄位標準化到 0-1 範圍，組成 10 維向量
    
    Args:
        row: DataFrame 的一行資料
        
    Returns:
        list: 標準化後的 10 維數值向量
    """
    return [
        row['danceability'],
        row['energy'],
        row['valence'],
        row['tempo'] / 200,  # tempo 約 0-200
        row['acousticness'],
        row['speechiness'],
        row['instrumentalness'],
        row['liveness'],
        (row['loudness'] + 60) / 60,  # loudness 約 -60 到 0
        row['popularity'] / 100
    ]


def load_dataset(filepath):
    """
    讀取 Spotify 資料集
    
    Args:
        filepath: CSV 檔案路徑
        
    Returns:
        pd.DataFrame: 資料集
    """
    df = pd.read_csv(filepath)
    print(f"✓ 資料筆數: {len(df)}")
    print(f"✓ 欄位: {df.columns.tolist()}")
    return df

# ============================================================================
# ChromaDB 操作函數（本機版本）
# ============================================================================
def initialize_chromadb(db_path):
    """
    初始化 ChromaDB 客戶端並載入 collections
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"❌ 找不到向量資料庫資料夾: {db_path}")

    client = chromadb.PersistentClient(path=db_path)

    collections = {
        "artist": client.get_collection("artist"),
        "track": client.get_collection("track"),
        "genre": client.get_collection("genre"),
        "combined": client.get_collection("combined"),
        "numeric": client.get_collection("numeric")
    }

    print("\n✓ ChromaDB 載入成功（本機模式）")
    for name, col in collections.items():
        print(f"  - {name}: {col.count()} 筆資料")

    return client, collections



def build_vector_database(df, collections, batch_size=500):
    """
    建立向量資料庫（首次建立時使用）
    
    Args:
        df: 資料集
        collections: ChromaDB collections
        batch_size: 批次大小
    """
    print(f"開始建立向量資料庫（完整 {len(df)} 筆）...")
    print("預估時間：約 2 小時\n")
    
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(df), batch_size), total=total_batches, desc="建立中"):
        batch = df.iloc[i:i+batch_size]
        
        # 準備 IDs
        ids = [str(idx) for idx in batch.index]
        
        # 準備文字資料
        artists = batch['artists'].fillna('').tolist()
        tracks = batch['track_name'].fillna('').tolist()
        genres = batch['track_genre'].fillna('').tolist()
        combined = [f"ARTIST: {a} TRACK: {t} GENRE: {g}" 
                   for a, t, g in zip(artists, tracks, genres)]
        
        # 生成 embeddings
        artist_emb = embedding_model.encode(artists).tolist()
        track_emb = embedding_model.encode(tracks).tolist()
        genre_emb = embedding_model.encode(genres).tolist()
        combined_emb = embedding_model.encode(combined).tolist()
        numeric_emb = [normalize_numeric(row) for _, row in batch.iterrows()]
        
        # 準備 metadata
        metadatas = batch[[
            'track_id', 'artists', 'track_name', 'album_name', 'track_genre',
            'popularity', 'danceability', 'energy', 'valence', 'tempo',
            'acousticness', 'speechiness', 'instrumentalness', 'liveness', 'loudness'
        ]].to_dict('records')
        
        # 寫入各 collection
        collections['artist'].add(ids=ids, embeddings=artist_emb, metadatas=metadatas)
        collections['track'].add(ids=ids, embeddings=track_emb, metadatas=metadatas)
        collections['genre'].add(ids=ids, embeddings=genre_emb, metadatas=metadatas)
        collections['combined'].add(ids=ids, embeddings=combined_emb, metadatas=metadatas)
        collections['numeric'].add(ids=ids, embeddings=numeric_emb, metadatas=metadatas)
    
    print(f"\n✓ 建立完成，每個 Collection 各 {collections['combined'].count()} 筆")


# ============================================================================
# 檢索函數
# ============================================================================
def search_by_artist(query, n=50):
    """用藝人名稱搜尋"""
    emb = embedding_model.encode(query).tolist()
    return collections['artist'].query(query_embeddings=[emb], n_results=n)


def search_by_track(query, n=50):
    """用歌曲名稱搜尋"""
    emb = embedding_model.encode(query).tolist()
    return collections['track'].query(query_embeddings=[emb], n_results=n)


def search_by_genre(query, n=50):
    """
    用類型搜尋 - 優先精確匹配，失敗則用向量檢索
    
    Args:
        query: 類型查詢
        n: 返回數量
        
    Returns:
        檢索結果
    """
    # 嘗試精確匹配
    try:
        results = collections['genre'].get(
            where={"track_genre": query.lower()},
            limit=n
        )
        if results['ids']:
            return {'metadatas': [results['metadatas']], 'ids': [results['ids']]}
    except:
        pass
    
    # 回退到向量檢索
    emb = embedding_model.encode(query).tolist()
    return collections['genre'].query(query_embeddings=[emb], n_results=n)


def search_combined(query, n=50):
    """用組合文字搜尋"""
    emb = embedding_model.encode(query).tolist()
    return collections['combined'].query(query_embeddings=[emb], n_results=n)


def search_by_numeric(danceability=0.5, energy=0.5, valence=0.5, tempo=120,
                      acousticness=0.5, speechiness=0.5, instrumentalness=0.5,
                      liveness=0.5, loudness=-10, popularity=50, n=50):
    """用數值特徵向量搜尋"""
    numeric_vector = [
        danceability, energy, valence, tempo/200,
        acousticness, speechiness, instrumentalness, liveness,
        (loudness+60)/60, popularity/100
    ]
    return collections['numeric'].query(query_embeddings=[numeric_vector], n_results=n)


# ============================================================================
# Claude API 互動函數
# ============================================================================
def analyze_intent(user_query):
    """
    使用 Claude 分析使用者查詢意圖
    
    Args:
        user_query: 使用者查詢字串
        
    Returns:
        dict: 包含 intent, search_text, numeric_filters 的字典
    """
    prompt = f"""分析這個音樂查詢的意圖,輸出 JSON 格式：

查詢：「{user_query}」

請輸出：
{{
    "intent": "artist/track/genre/mood/numeric",
    "search_text": "用於檢索的關鍵字（英文）",
    "numeric_filters": {{
        "tempo_min": null 或數字,
        "tempo_max": null 或數字,
        "energy_min": null 或 0-1,
        "energy_max": null 或 0-1,
        "danceability_min": null 或 0-1,
        "valence_min": null 或 0-1,
        "valence_max": null 或 0-1
    }}
}}

意圖判斷：
- artist：找特定歌手
- track：找特定歌曲
- genre：找特定類型（如 pop, rock, jazz, j-pop）
- mood：找情境/心情（如 放鬆、運動、悲傷）
- numeric：找特定音樂特性

只輸出 JSON，不要其他文字。"""

    try:
        response = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = response.content[0].text
        result = result.replace("```json", "").replace("```", "").strip()
        return json.loads(result)
    except Exception as e:
        print(f"意圖分析失敗: {e}")
        return {
            "intent": "combined", 
            "search_text": user_query, 
            "numeric_filters": {}
        }


def analyze_intent_with_context(user_query, session_history):
    """
    帶對話歷史的意圖分析（用於多輪對話）
    
    Args:
        user_query: 使用者查詢
        session_history: 對話歷史
        
    Returns:
        dict: 意圖分析結果
    """
    # 構建歷史文字
    history_text = ""
    if session_history:
        history_text = "之前的對話：\n"
        for h in session_history[-6:]:  # 只保留最近 6 輪
            history_text += f"使用者：{h['user']}\n"
            history_text += f"推薦類型：{h['genre']}\n"
            history_text += f"推薦歌曲：{h['songs']}\n\n"
    
    prompt = f"""{history_text}

現在使用者說：「{user_query}」

分析這個音樂查詢的意圖，輸出 JSON 格式：
{{
    "intent": "artist/track/genre/mood/numeric",
    "search_text": "用於檢索的關鍵字（英文）",
    "genre_filter": "精確的類型名稱，如果有指定的話（如 j-pop, rock, pop, jazz 等），否則為 null",
    "numeric_filters": {{
        "tempo_min": null 或數字,
        "tempo_max": null 或數字,
        "energy_min": null 或 0-1,
        "energy_max": null 或 0-1,
        "danceability_min": null 或 0-1,
        "valence_min": null 或 0-1,
        "valence_max": null 或 0-1
    }}
}}

重要提示：
- 日本流行歌 = j-pop（不是 j-dance）
- 韓國流行歌 = k-pop
- 如果使用者說「類似的」「再給我」「換一些」，請參考之前對話的類型和風格
- 如果之前推薦的是 j-pop，「類似的」也應該是 j-pop

只輸出 JSON，不要其他文字。"""

    try:
        response = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = response.content[0].text
        result = result.replace("```json", "").replace("```", "").strip()
        return json.loads(result)
    except Exception as e:
        print(f"⚠️ 意圖分析失敗: {e}")
        return {
            "intent": "combined",
            "search_text": user_query,
            "genre_filter": None,
            "numeric_filters": {}
        }


def generate_recommendation(user_query, songs):
    """
    使用 Claude 生成推薦說明
    
    Args:
        user_query: 使用者查詢
        songs: 推薦的歌曲列表
        
    Returns:
        str: 推薦說明文字
    """
    songs_text = "\n".join([
        f"{i+1}. {s['track_name']} - {s['artists']} "
        f"(類型:{s['track_genre']}, 能量:{s['energy']:.2f}, 節奏:{s['tempo']:.0f}BPM)"
        for i, s in enumerate(songs[:10])
    ])
    
    prompt = f"""使用者查詢：「{user_query}」

找到的歌曲：
{songs_text}

請用繁體中文簡短說明（2-3句）為什麼推薦這些歌"""

    try:
        response = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"⚠️ 生成推薦說明失敗: {e}")
        return "推薦歌曲如上。"


# ============================================================================
# 資料處理函數
# ============================================================================
def apply_numeric_filter(results, filters):
    """
    根據數值條件過濾候選歌曲
    
    Args:
        results: 檢索結果
        filters: 數值過濾條件字典
        
    Returns:
        list: 過濾後的歌曲列表
    """
    filtered = []
    
    for meta in results['metadatas'][0]:
        passed = True
        
        # 檢查各項條件
        if filters.get('tempo_min') and meta['tempo'] < filters['tempo_min']:
            passed = False
        if filters.get('tempo_max') and meta['tempo'] > filters['tempo_max']:
            passed = False
        if filters.get('energy_min') and meta['energy'] < filters['energy_min']:
            passed = False
        if filters.get('energy_max') and meta['energy'] > filters['energy_max']:
            passed = False
        if filters.get('danceability_min') and meta['danceability'] < filters['danceability_min']:
            passed = False
        if filters.get('valence_min') and meta['valence'] < filters['valence_min']:
            passed = False
        if filters.get('valence_max') and meta['valence'] > filters['valence_max']:
            passed = False
        
        if passed:
            filtered.append(meta)
    
    return filtered


def remove_duplicates(songs):
    """
    去除重複歌曲
    
    Args:
        songs: 歌曲列表
        
    Returns:
        list: 去重後的歌曲列表
    """
    seen = set()
    unique = []
    
    for s in songs:
        key = f"{s['track_name']}_{s['artists']}"
        if key not in seen:
            seen.add(key)
            unique.append(s)
    
    return unique


# ============================================================================
# Log 記錄函數
# ============================================================================
def write_log(user_query, intent_result, results_count, filtered_count, 
              final_songs, recommendation):
    """
    記錄查詢日誌
    
    Args:
        user_query: 使用者查詢
        intent_result: 意圖分析結果
        results_count: 候選數量
        filtered_count: 過濾後數量
        final_songs: 最終推薦歌曲
        recommendation: 推薦說明
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = f"""
{'='*60}
時間: {timestamp}
查詢: {user_query}
意圖: {intent_result['intent']}
檢索關鍵字: {intent_result['search_text']}
數值過濾: {intent_result.get('numeric_filters', {})}
候選數量: {results_count}
過濾後數量: {filtered_count}
推薦歌曲:
"""
    for i, s in enumerate(final_songs):
        log_entry += f"  {i+1}. {s['track_name']} - {s['artists']}\n"
    
    log_entry += f"\nClaude 推薦說明:\n{recommendation}\n"
    
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(log_entry)
        print(f"📝 已記錄到 log")
    except Exception as e:
        print(f"⚠️ 寫入 log 失敗: {e}")


# ============================================================================
# 主推薦函數
# ============================================================================

def recommend(user_query):
    """
    主推薦函數（加入路由判斷）
    """
    print(f"🎵 查詢：「{user_query}」\n")
    
    # 🔥 Step 0.5: 路由決策（新增）
    print("Step 0.5: 路由決策...")
    decision = router.route_query(user_query)
    print(f"   查詢類型: {decision.query_type.value}")
    print(f"   使用 Clustering: {'是' if decision.use_clustering else '否'}")
    print(f"   信心分數: {decision.confidence:.2f}")
    print(f"   判斷理由: {decision.reasoning}")
    
    # 🔥 判斷是否需要 Clustering
    if decision.use_clustering and genre_clusters is not None:
        print("\n   → 啟動 Clustering 多步驟檢索")
        return recommend_with_clustering(user_query, decision)
    else:
        if decision.use_clustering and genre_clusters is None:
            print("\n   ⚠️ Clustering 模型未載入，降級為向量檢索")
        else:
            print("\n   → 使用標準向量檢索")
        # 使用原有邏輯
        return recommend_original(user_query)


def recommend_original(user_query):
    """
    主推薦函數（單次查詢）
    
    Args:
        user_query: 使用者查詢
        
    Returns:
        list: 推薦的歌曲列表
    """
    print(f"🎵 查詢：「{user_query}」\n")
    
    # Step 1: 意圖分析
    print("Step 1: 分析意圖...")
    intent_result = analyze_intent(user_query)
    print(f"   意圖: {intent_result['intent']}")
    print(f"   檢索關鍵字: {intent_result['search_text']}")
    
    # Step 2: 向量檢索
    print("\ntep 2: 向量檢索...")
    intent = intent_result['intent']
    search_text = intent_result['search_text']
    
    if intent == 'artist':
        results = search_by_artist(search_text, n=CANDIDATE_SIZE)
    elif intent == 'track':
        results = search_by_track(search_text, n=CANDIDATE_SIZE)
    elif intent == 'genre':
        results = search_by_genre(search_text, n=CANDIDATE_SIZE)
    else:
        results = search_combined(search_text, n=CANDIDATE_SIZE)
    
    results_count = len(results['metadatas'][0])
    print(f"   找到 {results_count} 首候選歌曲")
    
    # Step 3: 數值過濾
    print("\n Step 3: 數值過濾...")
    filters = intent_result.get('numeric_filters', {})
    filters = {k: v for k, v in filters.items() if v is not None}
    
    if filters:
        print(f"   過濾條件: {filters}")
        filtered_songs = apply_numeric_filter(results, filters)
        filtered_count = len(filtered_songs)
        print(f"   過濾後: {filtered_count} 首")
    else:
        print("   無數值過濾")
        filtered_songs = results['metadatas'][0]
        filtered_count = len(filtered_songs)
    
    # 如果結果太少，使用原始結果
    if len(filtered_songs) < 5:
        print("   結果太少，使用原始結果")
        filtered_songs = results['metadatas'][0]
        filtered_count = len(filtered_songs)
    
    # Step 3.5: 去重處理
    print("\n Step 3.5: 去重處理...")
    unique_songs = remove_duplicates(filtered_songs)
    duplicate_count = len(filtered_songs) - len(unique_songs)
    
    if duplicate_count > 0:
        print(f"移除了 {duplicate_count} 首重複歌曲")
    else:
        print(f"無重複歌曲 ✓")
    
    final_songs = unique_songs[:FINAL_RECOMMENDATION]
    
    # Step 4: 生成推薦
    print("\nStep 4: 生成推薦說明...")
    recommendation = generate_recommendation(user_query, final_songs)
    
    # Step 5: 寫入 log
    write_log(user_query, intent_result, results_count, 
              filtered_count, final_songs, recommendation)
    
    print("\n" + "="*50)
    print(recommendation)
    
    return final_songs

def recommend_with_clustering(user_query, decision):
    """
    使用 Clustering 的多步驟推薦
    完全基於 tracks_df (parquet)，不依賴 ChromaDB metadata
    
    Parameters:
    -----------
    user_query : str
        使用者查詢
    decision : RoutingDecision
        路由決策結果
    """
    
    print("\n🔄 Clustering 多步驟檢索...")
    print(f"   查詢類型: {decision.query_type.value}")
    
    query_type = decision.query_type
    extracted = decision.extracted_info
    
    # ============================================================
    # 路徑 1: 近似歌曲推薦
    # ============================================================
    if query_type == QueryType.SIMILARITY_TRACK:
        target_track = extracted.get('target_track', user_query)
        print(f"   → 目標歌曲: {target_track}")
        
        # Step 1: 用 ChromaDB 找到最相似的歌
        print("\n   [Step 1] 從向量資料庫找目標歌曲...")
        target_results = collections['track'].query(
            query_embeddings=[embedding_model.encode(target_track).tolist()],
            n_results=1
        )
        
        if not target_results['ids'][0]:
            print("   ⚠️ 找不到目標歌曲，降級到原邏輯")
            return recommend_original(user_query)
        
        target_track_id = target_results['metadatas'][0][0]['track_id']
        target_track_name = target_results['metadatas'][0][0]['track_name']
        print(f"   ✓ 找到: {target_track_name} (ID: {target_track_id})")
        
        # Step 2: 從 parquet 查詢 cluster 資訊
        print("\n   [Step 2] 查詢 cluster 資訊...")
        track_info = tracks_df[tracks_df['track_id'] == target_track_id]
        
        if track_info.empty:
            print("   ⚠️ parquet 中找不到此歌曲，降級到原邏輯")
            return recommend_original(user_query)
        
        track_info = track_info.iloc[0]
        genre = track_info['track_genre']
        sub_cluster = track_info['sub_cluster']
        hierarchical_id = track_info.get('hierarchical_id', f"{genre}_cluster_{sub_cluster}")
        
        print(f"   ✓ Cluster: {hierarchical_id}")
        print(f"      Genre: {genre}")
        print(f"      Sub-cluster: {sub_cluster}")
        
        # Step 3: 從 parquet 找出同 cluster 的所有歌
        print("\n   [Step 3] 篩選同 cluster 歌曲...")
        same_cluster = tracks_df[
            (tracks_df['track_genre'] == genre) &
            (tracks_df['sub_cluster'] == sub_cluster)
        ]
        
        print(f"   ✓ 同 cluster 歌曲數: {len(same_cluster)}")
        
        # 如果同 cluster 歌曲太少，擴展到相鄰 cluster
        if len(same_cluster) < 20:
            print(f"   ⚠️ 歌曲數太少，擴展到整個 {genre}")
            same_cluster = tracks_df[tracks_df['track_genre'] == genre]
            print(f"   ✓ 擴展後: {len(same_cluster)} 首")
        
        # Step 4: 在 ChromaDB 搜尋，但只從同 cluster 的歌中選
        print("\n   [Step 4] 在 cluster 內做向量檢索...")
        cluster_track_ids = set(same_cluster['track_id'].tolist())
        
        # 用目標歌曲做相似度搜尋
        all_results = collections['combined'].query(
            query_embeddings=[embedding_model.encode(target_track).tolist()],
            n_results=min(1000, len(cluster_track_ids) * 2)  # 多取一些
        )
        
        # Step 5: 過濾出同 cluster 的歌
        filtered_songs = []
        for meta in all_results['metadatas'][0]:
            if meta['track_id'] in cluster_track_ids:
                # 排除目標歌曲本身
                if meta['track_id'] != target_track_id:
                    filtered_songs.append(meta)
            
            # 取夠了就停止
            if len(filtered_songs) >= CANDIDATE_SIZE:
                break
        
        print(f"   ✓ 過濾後候選: {len(filtered_songs)} 首")
        
        # Step 6: 去重、取 Top K
        unique_songs = remove_duplicates(filtered_songs)
        final_songs = unique_songs[:FINAL_RECOMMENDATION]
        
        # Step 7: 生成推薦
        recommendation = generate_recommendation(user_query, final_songs)
        
        print("\n" + "="*50)
        print(recommendation)
        
        return final_songs
    
    
    # ============================================================
    # 路徑 2: 近似歌手推薦
    # ============================================================
    elif query_type == QueryType.SIMILARITY_ARTIST:
        target_artist = extracted.get('target_artist', user_query)
        print(f"   → 目標歌手: {target_artist}")
        
        # Step 1: 找到該歌手的所有歌
        print("\n   [Step 1] 找目標歌手的歌曲...")
        artist_results = collections['artist'].query(
            query_embeddings=[embedding_model.encode(target_artist).tolist()],
            n_results=30  # 取前 30 首該歌手的歌
        )
        
        if not artist_results['ids'][0]:
            print("   ⚠️ 找不到該歌手，降級到原邏輯")
            return recommend_original(user_query)
        
        artist_track_ids = [meta['track_id'] for meta in artist_results['metadatas'][0]]
        print(f"   ✓ 找到 {len(artist_track_ids)} 首該歌手的歌")
        
        # Step 2: 從 parquet 統計歌手的 cluster 分布
        print("\n   [Step 2] 統計歌手的 cluster 分布...")
        artist_tracks = tracks_df[tracks_df['track_id'].isin(artist_track_ids)]
        
        if artist_tracks.empty:
            print("   ⚠️ parquet 中找不到該歌手，降級到原邏輯")
            return recommend_original(user_query)
        
        # 統計每個 cluster 的歌曲數
        cluster_counts = artist_tracks.groupby(['track_genre', 'sub_cluster']).size()
        cluster_counts = cluster_counts.sort_values(ascending=False)
        
        print(f"   ✓ 歌手的 cluster 分布:")
        for (genre, sub_cluster), count in cluster_counts.head(5).items():
            print(f"      - {genre}_cluster_{sub_cluster}: {count} 首")
        
        # Step 3: 選擇前 2-3 個主要 clusters
        top_clusters = cluster_counts.head(3).index.tolist()
        
        # Step 4: 從這些 clusters 推薦其他歌手的歌
        print("\n   [Step 3] 從主要 clusters 推薦其他歌手...")
        cluster_songs = tracks_df[
            tracks_df.apply(
                lambda row: (row['track_genre'], row['sub_cluster']) in top_clusters,
                axis=1
            )
        ]
        
        # 排除原歌手的歌
        target_artist_lower = target_artist.lower()
        cluster_songs = cluster_songs[
            ~cluster_songs['artists'].str.lower().str.contains(target_artist_lower, na=False)
        ]
        
        print(f"   ✓ 候選歌曲數: {len(cluster_songs)}")
        
        # Step 5: 在 ChromaDB 搜尋，從候選中選擇
        cluster_track_ids = set(cluster_songs['track_id'].tolist())
        
        all_results = collections['combined'].query(
            query_embeddings=[embedding_model.encode(target_artist).tolist()],
            n_results=min(1000, len(cluster_track_ids) * 2)
        )
        
        # 過濾
        filtered_songs = []
        seen_artists = set()
        
        for meta in all_results['metadatas'][0]:
            if meta['track_id'] in cluster_track_ids:
                artist = meta['artists']
                
                # 每個歌手最多 2 首（確保多樣性）
                if seen_artists.count(artist) < 2:
                    filtered_songs.append(meta)
                    seen_artists.add(artist)
            
            if len(filtered_songs) >= CANDIDATE_SIZE:
                break
        
        print(f"   ✓ 過濾後候選: {len(filtered_songs)} 首")
        print(f"   ✓ 涵蓋 {len(set(seen_artists))} 位不同歌手")
        
        # Step 6: 去重、取 Top K
        unique_songs = remove_duplicates(filtered_songs)
        final_songs = unique_songs[:FINAL_RECOMMENDATION]
        
        # Step 7: 生成推薦
        recommendation = generate_recommendation(user_query, final_songs)
        
        print("\n" + "="*50)
        print(recommendation)
        
        return final_songs
    
    
    # ============================================================
    # 路徑 3: 探索同類型音樂
    # ============================================================
    elif query_type == QueryType.CLUSTER_EXPLORATION:
        print("   → 探索模式（待實作）")
        print("   ⚠️ 暫時降級到原邏輯")
        return recommend_original(user_query)
    
    
    # ============================================================
    # 其他：降級到原邏輯
    # ============================================================
    else:
        print("   → 未知查詢類型，降級到原邏輯")
        return recommend_original(user_query)

def chat_recommend(user_message, session_id):
    """
    聊天式推薦（多輪對話）
    
    Args:
        user_message: 使用者訊息
        session_id: 對話 session ID
        
    Returns:
        str: 推薦說明文字
    """
    # 初始化 session
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    session_history = chat_sessions[session_id]
    
    # 收集已推薦過的歌曲
    recommended_songs = set()
    for h in session_history:
        for song in h['songs'].split(', '):
            recommended_songs.add(song.strip())
    
    # 意圖分析（帶歷史）
    intent_result = analyze_intent_with_context(user_message, session_history)
    
    intent = intent_result['intent']
    search_text = intent_result['search_text']
    
    # 向量檢索
    if intent == 'artist':
        results = search_by_artist(search_text, n=CANDIDATE_SIZE)
    elif intent == 'track':
        results = search_by_track(search_text, n=CANDIDATE_SIZE)
    elif intent == 'genre':
        results = search_by_genre(search_text, n=CANDIDATE_SIZE)
    else:
        results = search_combined(search_text, n=CANDIDATE_SIZE)
    
    # 如果有精確類型過濾
    genre_filter = intent_result.get('genre_filter')
    if genre_filter:
        results = search_by_genre(genre_filter, n=CANDIDATE_SIZE)
    
    # 數值過濾
    filters = intent_result.get('numeric_filters', {})
    filters = {k: v for k, v in filters.items() if v is not None}
    
    if filters:
        filtered_songs = apply_numeric_filter(results, filters)
    else:
        filtered_songs = results['metadatas'][0]
    
    if len(filtered_songs) < 5:
        filtered_songs = results['metadatas'][0]
    
    # 去重
    filtered_songs = remove_duplicates(filtered_songs)
    
    # 排除已推薦過的歌
    new_songs = [s for s in filtered_songs 
                 if s['track_name'] not in recommended_songs]
    
    # 如果新歌不夠，才用舊的補
    if len(new_songs) < FINAL_RECOMMENDATION:
        final_songs = new_songs + [
            s for s in filtered_songs 
            if s['track_name'] in recommended_songs
        ][:FINAL_RECOMMENDATION - len(new_songs)]
    else:
        final_songs = new_songs[:FINAL_RECOMMENDATION]
    
    # 生成推薦
    recommendation = generate_recommendation(user_message, final_songs)
    
    # 更新歷史
    songs_summary = ", ".join([s['track_name'] for s in final_songs])
    genre_summary = final_songs[0]['track_genre'] if final_songs else ""
    session_history.append({
        "user": user_message,
        "genre": genre_summary,
        "songs": songs_summary
    })
    chat_sessions[session_id] = session_history
    
    # 寫入 log
    write_log(user_message, intent_result, len(results['metadatas'][0]),
              len(filtered_songs), final_songs, recommendation)
    
    return recommendation


# ============================================================================
# Gradio 介面
# ============================================================================
def create_gradio_interface():
    """建立 Gradio 聊天介面"""
    
    def user_message(message, history):
        """處理使用者訊息"""
        if not message.strip():
            return "", history
        history.append((message, None))
        return "", history
    
    def bot_response(history, session_state):
        """生成機器人回應"""
        if session_state is None:
            session_state = str(uuid.uuid4())
        
        user_msg = history[-1][0]
        response = chat_recommend(user_msg, session_state)
        history[-1] = (user_msg, response)
        return history, session_state
    
    def clear_chat(session_state):
        """清除對話"""
        if session_state and session_state in chat_sessions:
            chat_sessions[session_state] = []
        return [], session_state
    
    # 建立介面
    with gr.Blocks(title="🎵 Spotify 音樂推薦系統", 
                   css="footer {display: none}") as demo:
        gr.Markdown("# 🎵 Spotify 音樂推薦系統")
        gr.Markdown("輸入你想找的音樂類型、歌手、心情或情境，我會推薦適合的歌曲！")
        
        session_state = gr.State(None)
        chatbot = gr.Chatbot(height=500, container=True, show_copy_button=True)
        
        with gr.Row(equal_height=True):
            msg = gr.Textbox(
                label="",
                placeholder="例如：給我日本流行歌",
                scale=6,
                container=False
            )
            submit = gr.Button("送出", scale=1, min_width=80)
        
        clear = gr.Button("清除對話", variant="secondary")
        
        # 綁定事件
        submit.click(
            user_message, [msg, chatbot], [msg, chatbot]
        ).then(
            bot_response, [chatbot, session_state], [chatbot, session_state]
        )
        
        msg.submit(
            user_message, [msg, chatbot], [msg, chatbot]
        ).then(
            bot_response, [chatbot, session_state], [chatbot, session_state]
        )
        
        clear.click(clear_chat, [session_state], [chatbot, session_state])
    
    return demo


# ============================================================================
# 主程式（本機 VS Code 版本）
# ============================================================================
def main():
    global chroma_client, collections, router, genre_clusters, tracks_df
    
    print("=" * 80)
    print("🎵 Spotify 音樂推薦系統 v2.0 - 整合 Clustering 路由")
    print("=" * 80)

    # ============================================================
    # Step 1: 確認檔案存在
    # ============================================================
    print("\nStep 1: 檢查必要檔案...")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ 找不到 dataset.csv：{DATA_PATH}")
        return

    if not os.path.exists(DB_PATH):
        print(f"❌ 找不到 ChromaDB：{DB_PATH}")
        return
    
    print("✓ 所有檔案存在")

    # ============================================================
    # Step 2: 載入向量資料庫
    # ============================================================
    print("\nStep 2: 載入向量資料庫...")
    chroma_client, collections = initialize_chromadb(DB_PATH)
    print("✓ 向量資料庫載入完成")

    # ============================================================
    # Step 3: 初始化路由器
    # ============================================================
    print("\nStep 3: 初始化查詢路由器...")
    router = QueryRouter()
    print("✓ 路由器初始化完成")

    # ============================================================
    # Step 4: 載入 Clustering 模型
    # ============================================================
    cluster_path = os.path.join(DRIVE_ROOT, "models/genre_clusters_v3.pkl")
    tracks_path = os.path.join(DRIVE_ROOT, "data/tracks_with_hierarchical_clusters_v3.parquet")
    
    print("\nStep 4: 載入 Clustering 模型...")
    
    if os.path.exists(tracks_path):
        tracks_df = pd.read_parquet(tracks_path)
        print(f"✓ 已載入 {len(tracks_df):,} 首歌曲資料")
        
        # 檢查必要欄位
        required_cols = ['track_id', 'track_genre', 'sub_cluster']
        missing_cols = [col for col in required_cols if col not in tracks_df.columns]
        
        if missing_cols:
            print(f"⚠️ 缺少欄位: {missing_cols}")
            tracks_df = None
        else:
            print(f"✓ 必要欄位檢查通過")
            
            # 如果沒有 hierarchical_id，就創建一個
            if 'hierarchical_id' not in tracks_df.columns:
                print("  → 創建 hierarchical_id 欄位...")
                tracks_df['hierarchical_id'] = (
                    tracks_df['track_genre'] + '_cluster_' + 
                    tracks_df['sub_cluster'].astype(str)
                )
                print("  ✓ hierarchical_id 創建完成")
            
            # 顯示 cluster 統計
            n_genres = tracks_df['track_genre'].nunique()
            n_clusters = tracks_df.groupby('track_genre')['sub_cluster'].nunique().sum()
            print(f"✓ 統計: {n_genres} 個 genres, {n_clusters} 個 sub-clusters")
    else:
        print(f"⚠️ 找不到 parquet 檔案：{tracks_path}")
        tracks_df = None
    
    if os.path.exists(cluster_path):
        genre_clusters = joblib.load(cluster_path)
        print(f"✓ 已載入 {len(genre_clusters)} 個 genre cluster 模型")
    else:
        print(f"⚠️ 找不到 cluster 模型：{cluster_path}")
        genre_clusters = None

    # ============================================================
    # 系統狀態總結
    # ============================================================
    print("\n" + "=" * 80)
    print("✅ 系統啟動完成！")
    print("=" * 80)
    print("\n系統功能狀態:")
    print(f"  ✓ 向量檢索: 可用")
    print(f"  {'✓' if tracks_df is not None else '✗'} Clustering 檢索: {'可用' if tracks_df is not None else '不可用'}")
    print(f"  ✓ 智能路由: 可用")
    print("=" * 80)

    # ============================================================
    # Step 5: 測試推薦功能
    # ============================================================
    print("\n🧪 開始測試推薦功能...\n")
    
    test_queries = [
        # 基礎查詢（不需要 clustering）
        "給我適合運動的音樂",
        "Taylor Swift 的歌",
        
        # Clustering 查詢（需要 clustering）
        "類似 Bohemian Rhapsody 的歌",
        "推薦像 Adele 的歌手",
    ]
    
    for idx, query in enumerate(test_queries, 1):
        print("\n" + "=" * 80)
        print(f"[測試 {idx}/{len(test_queries)}] 查詢: 「{query}」")
        print("=" * 80)
        
        try:
            songs = recommend(query)
            
            if songs:
                print(f"\n✅ 推薦結果（共 {len(songs)} 首）:")
                for i, song in enumerate(songs[:5], 1):
                    genre = song.get('track_genre', 'Unknown')
                    print(f"  {i}. {song['track_name']} - {song['artists']} [{genre}]")
                
                if len(songs) > 5:
                    print(f"  ... 還有 {len(songs) - 5} 首")
            else:
                print("\n⚠️ 沒有找到推薦結果")
        
        except Exception as e:
            print(f"\n❌ 錯誤: {e}")
            import traceback
            traceback.print_exc()
        
        # 避免 API rate limit
        if idx < len(test_queries):
            print("\n⏳ 等待 2 秒...")
            time.sleep(2)
    
    # ============================================================
    # 完成
    # ============================================================
    print("\n" + "=" * 80)
    print("🎉 測試完成！")
    print("=" * 80)
    
    # 提供互動式測試選項
    print("\n💡 提示:")
    print("  - 所有測試已完成")
    print("  - 如需互動測試，請使用 Gradio 介面")
    print("  - 如需調整 clustering，請修改 parquet 檔案後重新載入")


if __name__ == "__main__":
    main()