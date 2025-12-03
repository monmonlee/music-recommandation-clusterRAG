import os
import json
import shutil
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import BaseRetriever, Document
from langchain_anthropic import ChatAnthropic
from typing import List, Any
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
import pandas as pd



def process_spotify_csv(csv_path, limit=None):
    """
    讀取 Spotify CSV 並轉換成文字描述
    """
    df = pd.read_csv(csv_path)


    if limit:
        df = df.head(limit)
        print(f"限制處理前 {limit} 筆資料")
    
    documents = []
    
    for idx, row in df.iterrows():
        # 方法 1：分欄位描述（推薦）
        text = f"""
        ARTIST: {row['artists']}
        TRACK: {row['track_name']}
        ALBUM: {row['album_name']}
        GENRE: {row['track_genre']}
        
        音樂特性：
        - 節奏感 (Danceability): {row['danceability']:.2f}
        - 能量 (Energy): {row['energy']:.2f}
        - 響度 (Loudness): {row['loudness']:.1f} dB
        - 速度 (Tempo): {row['tempo']:.0f} BPM
        - 正面情緒 (Valence): {row['valence']:.2f}
        - 熱門度 (Popularity): {row['popularity']}/100
        """.strip()
        
        # 保留原始數值作為 metadata（重要！）
        metadata = {
            'track_id': row['track_id'],
            'artists': row['artists'],
            'track_name': row['track_name'],
            'album_name': row['album_name'],
            'track_genre': row['track_genre'],
            'popularity': float(row['popularity']),
            'danceability': float(row['danceability']),
            'energy': float(row['energy']),
            'loudness': float(row['loudness']),
            'tempo': float(row['tempo']),
            'valence': float(row['valence']),
            'speechiness': float(row['speechiness']),
            'acousticness': float(row['acousticness']),
            'instrumentalness': float(row['instrumentalness']),
        }
        
        documents.append(Document(
            page_content=text,
            metadata=metadata
        ))
    
    return documents

def setup_environment_with_csv(csv_path,  limit=None):
    """
    完整設定：載入 CSV + 建立向量資料庫 + 初始化 Claude
    """
    _ = load_dotenv(find_dotenv())
    
    # 1. 初始化 Claude LLM
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514", 
        temperature=0,
        api_key=os.environ['ANTHROPIC_API_KEY']
    )
    
    # 2. 初始化 Embedding（用 OpenAI）
    embeddings = OpenAIEmbeddings(
        api_key=os.environ['OPENAI_API_KEY']
    )
    
    # 3. 處理 CSV
    print("處理 Spotify CSV...")
    documents = process_spotify_csv(csv_path, limit=limit)
    print(f"成功處理 {len(documents)} 首歌曲")
    
    # 4. 建立向量資料庫
    print("建立向量資料庫...")
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./spotify_chroma_db"
    )
    print("向量資料庫建立完成！")
    
    return llm, vectordb, documents



def rag_music_recommendation(llm, vectordb, user_query):
    """
    混合檢索：向量相似度 + 數值過濾
    """
    
    # Step 1: Claude 分析意圖並提取數值條件
    intent_prompt = f"""
    使用者查詢：「{user_query}」
    
    請分析這個查詢，並以 JSON 格式輸出：
    {{
      "intent": "查詢意圖類型（artist/genre/mood/activity）",
      "keywords": ["關鍵字1", "關鍵字2"],
      "numeric_filters": {{
        "tempo_min": 數字或null,
        "energy_min": 數字或null,
        "danceability_min": 數字或null,
        "valence_min": 數字或null,
        "acousticness_max": 數字或null
      }}
    }}
    
    只輸出 JSON，不要其他文字。
    """
    
    intent_response = llm.invoke(intent_prompt)
    print(f"🤖 意圖分析：\n{intent_response.content}\n")
    
    # 解析 JSON（處理可能的 markdown 包裹）
    try:
        intent_text = intent_response.content.strip()
        if intent_text.startswith("```json"):
            intent_text = intent_text.split("```json")[1].split("```")[0].strip()
        intent_data = json.loads(intent_text)
    except:
        intent_data = {"numeric_filters": {}}
    
    # Step 2: 向量檢索（先找候選歌曲）
    candidates = vectordb.similarity_search(user_query, k=50)  # 多找一些候選
    print(f"📊 向量檢索找到 {len(candidates)} 首候選歌曲")
    
    # Step 3: 數值過濾
    filters = intent_data.get("numeric_filters", {})
    filtered_results = []
    
    for doc in candidates:
        # 檢查是否符合所有數值條件
        if filters.get("tempo_min") and doc.metadata['tempo'] < filters['tempo_min']:
            continue
        if filters.get("energy_min") and doc.metadata['energy'] < filters['energy_min']:
            continue
        if filters.get("danceability_min") and doc.metadata['danceability'] < filters['danceability_min']:
            continue
        if filters.get("valence_min") and doc.metadata['valence'] < filters['valence_min']:
            continue
        if filters.get("acousticness_max") and doc.metadata['acousticness'] > filters['acousticness_max']:
            continue
        
        filtered_results.append(doc)
        
        if len(filtered_results) >= 10:  # 只要 10 首就夠
            break
    
    print(f"✅ 數值過濾後剩下 {len(filtered_results)} 首歌曲")
    
    # Step 4: 組織檢索結果
    final_songs = filtered_results[:5]
    context = "\n\n".join([
        f"歌曲 {i+1}: {doc.metadata['track_name']} - {doc.metadata['artists']}\n"
        f"類型: {doc.metadata['track_genre']}\n"
        f"特性: 能量 {doc.metadata['energy']:.2f}, 節奏感 {doc.metadata['danceability']:.2f}, "
        f"速度 {doc.metadata['tempo']:.0f} BPM, 正面情緒 {doc.metadata['valence']:.2f}"
        for i, doc in enumerate(final_songs)
    ])
    
    # Step 5: Claude 生成推薦說明
    recommendation_prompt = f"""
    使用者問：「{user_query}」
    
    根據以下檢索到的歌曲，生成推薦理由：
    
    {context}
    
    請用友善、簡潔的方式推薦這些歌曲，說明為什麼適合使用者的需求。
    """
    
    recommendation = llm.invoke(recommendation_prompt)
    
    return {
        'songs': final_songs,
        'explanation': recommendation.content,
        'intent': intent_data
    }


if __name__ == "__main__":

    '''for building vectorDB'''

    # db_path = "./spotify_chroma_db"
    # if os.path.exists(db_path):
    #     shutil.rmtree(db_path)
    #     print("✅ 已刪除舊的向量資料庫")


    # llm, vectordb, docs = setup_environment_with_csv(
    #     "/Users/mangtinglee/Desktop/UT/data mining/期末/dataset.csv",
    #     limit=5000,
    # )

    '''for reading only'''
    _ = load_dotenv(find_dotenv())
    embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        api_key=os.environ['ANTHROPIC_API_KEY']
    )
    
    # 載入已存在的向量資料庫
    vectordb = Chroma(
        persist_directory="./spotify_chroma_db",
        embedding_function=embeddings
    )

    print(f"✅ 載入向量資料庫完成，共 {vectordb._collection.count()} 首歌")


    # ===== 使用新的混合檢索函數 =====
    result = rag_music_recommendation(
        llm, 
        vectordb, 
        "我想找適合清晨散步用的歌"
    )
    
    print("\n🎵 推薦結果：")
    print(result['explanation'])
    
    print("\n📋 推薦歌曲列表：")
    for i, doc in enumerate(result['songs'], 1):
        print(f"\n{i}. {doc.metadata['track_name']} - {doc.metadata['artists']}")
        print(f"   類型: {doc.metadata['track_genre']}")
        print(f"   能量: {doc.metadata['energy']:.2f} | 節奏: {doc.metadata['danceability']:.2f}")
        print(f"   速度: {doc.metadata['tempo']:.0f} BPM | 聲學性: {doc.metadata['acousticness']:.2f}")
