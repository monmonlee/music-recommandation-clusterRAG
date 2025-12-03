# query_router.py (獨立檔案)
"""
Query Router - 智能查詢路由系統
"""

import json
from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass
import anthropic
import os
from dotenv import load_dotenv, find_dotenv

class QueryType(Enum):
    """查詢類型定義"""
    CONTEXT_SEARCH = "context_search"      # 情境搜尋
    DIRECT_SEARCH = "direct_search"        # 直接查詢
    SIMILARITY_TRACK = "similarity_track"  # 近似歌曲
    SIMILARITY_ARTIST = "similarity_artist" # 近似歌手
    CLUSTER_EXPLORATION = "cluster_exploration"

@dataclass
class RoutingDecision:
    """路由決策結果"""
    query_type: QueryType
    use_clustering: bool
    extracted_info: Dict
    reasoning: str
    confidence: float

class QueryRouter:
    def __init__(self, api_key: Optional[str] = None):
        _ = load_dotenv(find_dotenv())
        self.client = anthropic.Anthropic(
            api_key=os.environ['ANTHROPIC_API_KEY']
        )
        self.model = "claude-3-5-haiku-20241022"
    
    def route_query(self, user_query: str) -> RoutingDecision:
        """分析查詢並決定路由"""
        prompt = self._build_routing_prompt(user_query)
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return self._parse_llm_response(response.content[0].text)
            
        except Exception as e:
            print(f"⚠️ 路由錯誤: {e}")
            return RoutingDecision(
                query_type=QueryType.DIRECT_SEARCH,
                use_clustering=False,
                extracted_info={"original_query": user_query},
                reasoning="路由器錯誤，使用預設策略",
                confidence=0.0
            )
    
    def _build_routing_prompt(self, user_query: str) -> str:
        return f"""你是音樂推薦系統的查詢分類專家。請分析使用者查詢並判斷檢索策略。

## 查詢類型定義

1. **context_search** (情境搜尋)
   - 特徵: 描述心情、場景、氛圍、活動
   - 範例: 「適合運動的音樂」「想聽悲傷的歌」「早晨咖啡廳」
   - 策略: 向量語義搜尋(不需 clustering)

2. **direct_search** (直接查詢)
   - 特徵: 明確歌名、歌手、專輯
   - 範例: 「Taylor Swift」「播放 Shake It Off」「給我日本流行歌」
   - 策略: 向量精確匹配(不需 clustering)

3. **similarity_track** (近似推薦-歌曲)
   - 特徵: 想找「類似某首歌」的音樂
   - 關鍵字: 類似、像、相似、同風格 + 歌名
   - 範例: 「類似 Bohemian Rhapsody 的歌」「跟這首風格接近的」
   - 策略: **需要 clustering**

4. **similarity_artist** (近似推薦-歌手)
   - 特徵: 想找「類似某歌手」的音樂
   - 關鍵字: 類似、像、風格接近 + 歌手名
   - 範例: 「推薦類似 Adele 的歌手」「跟 Ed Sheeran 風格接近的」
   - 策略: **需要 clustering**

5. **cluster_exploration** (探索同類型)
   - 特徵: 探索某風格、流派
   - 關鍵字: 同類型、這一群、相同風格、更多這種
   - 範例: 「更多這種風格的歌」「同類型的音樂」
   - 策略: **需要 clustering**

## 使用者查詢
「{user_query}」

## 輸出格式(必須是有效 JSON)

{{
  "query_type": "選擇上述五種類型之一",
  "use_clustering": true/false,
  "extracted_info": {{
    "target_track": "若是 similarity_track，提取歌名，否則 null",
    "target_artist": "若是 similarity_artist，提取歌手名，否則 null",
    "context_keywords": ["若是 context_search，提取關鍵情境詞"],
    "original_query": "{user_query}"
  }},
  "reasoning": "簡短說明判斷理由(1-2句)",
  "confidence": 0.0-1.0 的信心分數
}}

請直接輸出 JSON，不要有其他說明文字。"""

    def _parse_llm_response(self, llm_output: str) -> RoutingDecision:
        """解析 LLM JSON 回應"""
        try:
            cleaned = llm_output.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            result = json.loads(cleaned)
            
            return RoutingDecision(
                query_type=QueryType(result["query_type"]),
                use_clustering=result["use_clustering"],
                extracted_info=result["extracted_info"],
                reasoning=result["reasoning"],
                confidence=result["confidence"]
            )
        except Exception as e:
            print(f"❌ 解析失敗: {e}\n原始回應: {llm_output}")
            raise


# 測試用主程式
if __name__ == "__main__":
    router = QueryRouter()
    
    test_queries = [
        "適合運動的音樂",
        "Taylor Swift 的歌",
        "類似 Bohemian Rhapsody 的歌",
        "推薦像 Adele 的歌手",
        "給我日本流行歌",
        "更多這種風格的 jazz"
    ]
    
    print("=" * 80)
    print("🎯 路由器測試")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[測試 {i}] 查詢: 「{query}」")
        print("-" * 80)
        
        decision = router.route_query(query)
        
        print(f"✓ 類型: {decision.query_type.value}")
        print(f"✓ Clustering: {'是' if decision.use_clustering else '否'}")
        print(f"✓ 信心: {decision.confidence:.2f}")
        print(f"✓ 理由: {decision.reasoning}")
        print(f"✓ 提取資訊: {json.dumps(decision.extracted_info, ensure_ascii=False, indent=2)}")