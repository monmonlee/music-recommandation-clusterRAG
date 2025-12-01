## 專案概述（Project Overview）

本專案利用 Kaggle 的 Spotify Dataset，打造一個結合大型語言模型（LLM）的音樂推薦系統。  
系統核心包含以下能力：

- 自然語言理解  
- 候選歌曲的向量檢索  
- 依音樂特徵進行數值過濾  
- 基於歌手／曲目／曲風 embedding 的集群相似度模型  
- 依不同查詢類型進行多步推理與推薦  

---

## 基線檢索流程（Baseline Retrieval Pipeline, v1）

### 1. LLM 查詢理解  
使用者輸入的自然語言查詢會由 LLM 分析與結構化，產生一組一致的 JSON 數值化特徵，包括：

- 情緒與能量（mood / energy）  
- 節奏偏好（tempo）  
- 曲風意圖（genre intent）  
- 相似歌手或相似曲目意圖  
- 情境描述（contextual description）

---

### 2. 向量檢索：取得 50 首候選歌曲  
將使用者查詢向量化後，對整個資料庫進行相似度比對，並取得前 50 首候選歌曲。  
每首歌曲會建立一段結構化描述文字，包括：
```bash

    ARTIST: {artists}
    TRACK: {track_name}
    ALBUM: {album_name}
    GENRE: {track_genre}

    音樂特性：

    Danceability: {danceability}

    Energy: {energy}

    Loudness: {loudness}

    Tempo: {tempo}

    Valence: {valence}

    Popularity: {popularity}
```

---

### 3. 數值過濾與排序  
根據 LLM 產生的目標數值特徵（如 tempo 範圍、valence、energy、genre 權重等），  
對 50 首候選歌曲進行二次排序，選出最終前 10 首。

---

### 4. 推薦結果生成  
將候選歌曲的結構化資料與排序分數整合，產生最終 LLM 回應，包含：

- 推薦曲目  
- 推薦理由  
- 關鍵音樂特徵  
- 相似度判斷邏輯  

---

## 開發路線（Upcoming Features, v2–v3 Roadmap）

### 1. 建立評估基準  
為基線系統建立各類評估指標，包括：

- 向量檢索準確性  
- 數值特徵匹配程度  
- LLM 推論相關性  
- 使用者偏好對齊度  

---

### 2. Cluster-RAG 模組  
基線檢索對於下列「推薦類型」表現較弱：

- 類似歌手推薦  
- 類似曲目推薦  
- 類似曲風推薦  
- 曲目推薦歌手、或歌手推薦曲目  

因此將加入 Cluster-RAG：  
利用 K-Means / HDBSCAN 建立歌曲 embedding 的集群標籤，作為 LLM 的結構化知識，強化跨模態相似度推薦。

---

### 3. 集群評估（Cluster Evaluation）  
比較基線與 Cluster-RAG 的差異，包括：

- 集群內餘弦相似度  
- Silhouette Score  
- 曲風純度（genre purity）  
- LLM 主觀評估（pairwise ranking）

---

### 4. 多步推理之推薦引擎  
系統將支援兩大類推理流程：

| 查詢類型 | 處理方式 |
|----------|----------|
| 情境類（mood / activity / feeling） | 基線檢索（向量 + 數值過濾） |
| 查找類（搜尋特定資訊） | 基線檢索 |
| 推薦類（相似歌手／相似曲目） | Cluster-RAG + 相似度管線 |

---
