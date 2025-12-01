import os
import json
from dotenv import load_dotenv, find_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic

# 載入環境變數
load_dotenv(find_dotenv())

# 初始化（不需要重建資料庫）
embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0,
    api_key=os.environ['CLAUDE_API_KEY']
)

# 載入已存在的向量資料庫
vectordb = Chroma(
    persist_directory="./spotify_chroma_db",
    embedding_function=embeddings
)

print(f"✅ 載入向量資料庫完成，共 {vectordb._collection.count()} 首歌")

# ===== 加入你的新函數 =====
def rag_music_recommendation(llm, vectordb, user_query):
    # ... (前面給你的完整函數)
    pass

# ===== 測試 =====
if __name__ == "__main__":
    result = rag_music_recommendation(
        llm, 
        vectordb, 
        "我想找適合深夜讀書的安靜音樂"
    )
    
    print("\n🎵 推薦結果：")
    print(result['explanation'])