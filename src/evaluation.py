"""測試資料與評估指標"""
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
import pandas as pd
from collections import Counter
import numpy as np

# 從 rag_data.py 匯入檢索函數
from rag_data import rag_music_recommendation


class MusicRecommendationEvaluator:
    """
    音樂推薦系統評估器
    """
    
    def __init__(self, llm, vectordb):
        """
        初始化評估器
        
        參數:
            llm: Claude LLM 模型
            vectordb: Chroma 向量資料庫
        """
        self.llm = llm
        self.vectordb = vectordb
        self.results = []
    
    def load_test_queries(self, json_path):
        """
        從 JSON 載入測試查詢
        
        參數:
            json_path: 測試查詢 JSON 檔案路徑
            
        返回:
            所有查詢的列表
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 將所有類型的查詢展平成一個列表
        all_queries = []
        for category, queries in data.items():
            for query in queries:
                query['category'] = category
                all_queries.append(query)
        
        print(f"載入了 {len(all_queries)} 個測試查詢")
        return all_queries
    
    def run_single_test(self, test_case):
        """
        對單個查詢執行測試
        
        參數:
            test_case: 單個測試案例
            
        返回:
            測試結果字典
        """
        query_id = test_case['id']
        query = test_case['query']
        expected_features = test_case.get('expected_features', {})
        
        print(f"\n測試 [{query_id}]: {query}")
        
        # 記錄開始時間
        start_time = time.time()
        
        # 執行檢索（使用 rag_data.py 的函數）
        try:
            result = rag_music_recommendation(self.llm, self.vectordb, query)
            total_time = time.time() - start_time
            
            # 提取推薦結果
            recommendations = result['songs']
            intent_data = result['intent']
            explanation = result['explanation']
            
            # 計算評估指標
            metrics = self.calculate_metrics(recommendations, expected_features)
            
            # 組織結果
            test_result = {
                'query_id': query_id,
                'category': test_case['category'],
                'query': query,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_time': round(total_time, 2),
                'intent_data': intent_data,
                'recommendations': [
                    {
                        'track_name': doc.metadata['track_name'],
                        'artists': doc.metadata['artists'],
                        'track_genre': doc.metadata['track_genre'],
                        'energy': doc.metadata['energy'],
                        'danceability': doc.metadata['danceability'],
                        'valence': doc.metadata['valence'],
                        'tempo': doc.metadata['tempo'],
                        'acousticness': doc.metadata['acousticness'],
                        'speechiness': doc.metadata['speechiness'],
                        'popularity': doc.metadata['popularity']
                    }
                    for doc in recommendations
                ],
                'explanation': explanation,
                'metrics': metrics,
                'expected_features': expected_features,
                'notes': test_case.get('notes', '')
            }
            
            print(f"完成時間: {total_time:.2f} 秒")
            print(f"特徵匹配度: {metrics['feature_match_score']:.2f}")
            print(f"多樣性分數: {metrics['diversity_score']:.2f}")
            if metrics['has_duplicates']:
                print(f"警告: 發現重複推薦")
            
            return test_result
            
        except Exception as e:
            print(f"錯誤: {str(e)}")
            return {
                'query_id': query_id,
                'category': test_case['category'],
                'query': query,
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def calculate_metrics(self, recommendations, expected_features):
        """
        計算各種評估指標
        
        參數:
            recommendations: 推薦的歌曲列表（Document 物件）
            expected_features: 預期特徵字典
            
        返回:
            評估指標字典
        """
        metrics = {}
        
        # 1. 檢查重複
        metrics['has_duplicates'] = self.check_duplicates(recommendations)
        
        # 2. 計算多樣性
        diversity_metrics = self.calculate_diversity(recommendations)
        metrics.update(diversity_metrics)
        
        # 3. 計算特徵匹配度
        if expected_features:
            metrics['feature_match_score'] = self.check_feature_match(
                recommendations, expected_features
            )
        else:
            metrics['feature_match_score'] = None
        
        return metrics
    
    def check_duplicates(self, recommendations):
        """
        檢查是否有重複推薦
        
        參數:
            recommendations: 推薦的歌曲列表
            
        返回:
            是否有重複 (bool)
        """
        # 使用 (track_name, artists) 組合作為唯一識別
        song_ids = [
            (doc.metadata['track_name'], doc.metadata['artists'])
            for doc in recommendations
        ]
        
        return len(song_ids) != len(set(song_ids))
    
    def calculate_diversity(self, recommendations):
        """
        計算推薦結果的多樣性
        
        參數:
            recommendations: 推薦的歌曲列表
            
        返回:
            多樣性指標字典
        """
        if not recommendations:
            return {
                'diversity_score': 0,
                'artist_diversity': 0,
                'genre_diversity': 0,
                'feature_variance': 0
            }
        
        # 1. 藝人多樣性
        artists = [doc.metadata['artists'] for doc in recommendations]
        artist_diversity = len(set(artists)) / len(recommendations)
        
        # 2. 類型多樣性
        genres = [doc.metadata['track_genre'] for doc in recommendations]
        genre_diversity = len(set(genres)) / len(recommendations)
        
        # 3. 數值特徵變異係數（以 energy 為例）
        energies = [doc.metadata['energy'] for doc in recommendations]
        if len(energies) > 1:
            feature_variance = np.std(energies) / (np.mean(energies) + 1e-6)
        else:
            feature_variance = 0
        
        # 綜合多樣性分數（藝人和類型的平均）
        diversity_score = (artist_diversity + genre_diversity) / 2
        
        return {
            'diversity_score': round(diversity_score, 2),
            'artist_diversity': round(artist_diversity, 2),
            'genre_diversity': round(genre_diversity, 2),
            'feature_variance': round(feature_variance, 2)
        }
    
    def check_feature_match(self, recommendations, expected_features):
        """
        檢查推薦結果是否符合預期特徵範圍
        
        參數:
            recommendations: 推薦的歌曲列表
            expected_features: 預期特徵字典
            
        返回:
            特徵匹配度分數 (0.0-1.0)
        """
        if not recommendations or not expected_features:
            return None
        
        match_count = 0
        total_features = 0
        
        # 檢查數值範圍特徵
        numeric_features = ['energy', 'danceability', 'valence', 'tempo', 
                          'acousticness', 'speechiness', 'popularity']
        
        for feature in numeric_features:
            if feature in expected_features:
                expected_range = expected_features[feature]
                if isinstance(expected_range, list) and len(expected_range) == 2:
                    # 計算推薦歌曲該特徵的平均值
                    values = [doc.metadata[feature] for doc in recommendations]
                    avg_value = np.mean(values)
                    
                    # 檢查是否在預期範圍內
                    if expected_range[0] <= avg_value <= expected_range[1]:
                        match_count += 1
                    
                    total_features += 1
        
        # 檢查藝人匹配
        if 'artist_match' in expected_features:
            expected_artist = expected_features['artist_match']
            artists = [doc.metadata['artists'] for doc in recommendations]
            if all(expected_artist.lower() in artist.lower() for artist in artists):
                match_count += 1
            total_features += 1
        
        # 檢查類型匹配
        if 'genre_contains' in expected_features:
            expected_genres = expected_features['genre_contains']
            genres = [doc.metadata['track_genre'] for doc in recommendations]
            if any(any(eg in g for eg in expected_genres) for g in genres):
                match_count += 1
            total_features += 1
        
        if total_features == 0:
            return None
        
        return round(match_count / total_features, 2)
    
    def run_batch_evaluation(self, test_queries):
        """
        對所有查詢執行測試
        
        參數:
            test_queries: 測試查詢列表
            
        返回:
            所有測試結果的列表
        """
        results = []
        
        print(f"\n開始批次評估，共 {len(test_queries)} 個查詢...")
        print("=" * 60)
        
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n進度: {i}/{len(test_queries)}")
            result = self.run_single_test(test_case)
            results.append(result)
            time.sleep(0.5)  # 避免 API rate limit
        
        self.results = results
        return results
    
    def generate_statistics(self):
        """
        生成統計資料
        
        返回:
            統計資料字典
        """
        if not self.results:
            return {}
        
        # 過濾出成功的結果（沒有錯誤的）
        successful_results = [r for r in self.results if 'error' not in r]
        
        if not successful_results:
            return {'error': '沒有成功的測試結果'}
        
        # 總體統計
        total_tests = len(self.results)
        successful_tests = len(successful_results)
        
        # 時間統計
        times = [r['total_time'] for r in successful_results]
        avg_time = np.mean(times)
        
        # 重複率
        duplicate_count = sum(1 for r in successful_results 
                            if r['metrics']['has_duplicates'])
        duplicate_rate = duplicate_count / successful_tests
        
        # 特徵匹配度（只計算有預期特徵的）
        feature_scores = [r['metrics']['feature_match_score'] 
                         for r in successful_results 
                         if r['metrics']['feature_match_score'] is not None]
        avg_feature_match = np.mean(feature_scores) if feature_scores else None
        
        # 多樣性統計
        diversity_scores = [r['metrics']['diversity_score'] 
                          for r in successful_results]
        avg_diversity = np.mean(diversity_scores)
        
        artist_diversity_scores = [r['metrics']['artist_diversity'] 
                                  for r in successful_results]
        avg_artist_diversity = np.mean(artist_diversity_scores)
        
        genre_diversity_scores = [r['metrics']['genre_diversity'] 
                                 for r in successful_results]
        avg_genre_diversity = np.mean(genre_diversity_scores)
        
        # 按類別統計
        categories = {}
        for result in successful_results:
            category = result['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        category_stats = {}
        for category, results in categories.items():
            category_stats[category] = {
                'count': len(results),
                'avg_time': round(np.mean([r['total_time'] for r in results]), 2),
                'avg_diversity': round(np.mean([r['metrics']['diversity_score'] 
                                               for r in results]), 2),
                'duplicate_rate': sum(1 for r in results 
                                    if r['metrics']['has_duplicates']) / len(results)
            }
            
            # 特徵匹配度（如果有的話）
            cat_feature_scores = [r['metrics']['feature_match_score'] 
                                 for r in results 
                                 if r['metrics']['feature_match_score'] is not None]
            if cat_feature_scores:
                category_stats[category]['avg_feature_match'] = round(
                    np.mean(cat_feature_scores), 2
                )
        
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': round(successful_tests / total_tests, 2),
            'avg_time': round(avg_time, 2),
            'duplicate_rate': round(duplicate_rate, 2),
            'avg_feature_match': round(avg_feature_match, 2) if avg_feature_match else None,
            'avg_diversity': round(avg_diversity, 2),
            'avg_artist_diversity': round(avg_artist_diversity, 2),
            'avg_genre_diversity': round(avg_genre_diversity, 2),
            'category_stats': category_stats
        }
    
    def generate_evaluation_report(self, output_path='evaluation_report.txt'):
        """
        生成可讀的評估報告
        
        參數:
            output_path: 報告輸出路徑
        """
        stats = self.generate_statistics()
        
        if 'error' in stats:
            print(f"錯誤: {stats['error']}")
            return
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # 標題
            f.write("=" * 60 + "\n")
            f.write("音樂推薦系統 Baseline 評估報告\n")
            f.write(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            # 總體統計
            f.write("【總體統計】\n")
            f.write(f"- 總測試數: {stats['total_tests']} 個查詢\n")
            f.write(f"- 成功率: {stats['success_rate']*100:.1f}%\n")
            f.write(f"- 平均回應時間: {stats['avg_time']} 秒\n")
            f.write(f"- 重複推薦發生率: {stats['duplicate_rate']*100:.1f}%\n")
            if stats['avg_feature_match'] is not None:
                f.write(f"- 平均特徵匹配度: {stats['avg_feature_match']}\n")
            f.write("\n")
            
            # 品質指標
            f.write("【品質指標】\n")
            f.write(f"- 平均多樣性分數: {stats['avg_diversity']}\n")
            f.write(f"- 平均藝人多樣性: {stats['avg_artist_diversity']}\n")
            f.write(f"- 平均類型多樣性: {stats['avg_genre_diversity']}\n")
            f.write("\n")
            
            # 各類型查詢表現
            f.write("=" * 60 + "\n")
            f.write("【各類型查詢表現】\n")
            f.write("=" * 60 + "\n\n")
            
            for category, cat_stats in stats['category_stats'].items():
                f.write(f"{category} ({cat_stats['count']} 個):\n")
                f.write(f"- 平均回應時間: {cat_stats['avg_time']} 秒\n")
                f.write(f"- 平均多樣性: {cat_stats['avg_diversity']}\n")
                f.write(f"- 重複率: {cat_stats['duplicate_rate']*100:.1f}%")
                if cat_stats['duplicate_rate'] > 0:
                    f.write(" [警告]")
                f.write("\n")
                if 'avg_feature_match' in cat_stats:
                    f.write(f"- 平均特徵匹配度: {cat_stats['avg_feature_match']}\n")
                f.write("\n")
            
            # 問題案例
            f.write("=" * 60 + "\n")
            f.write("【問題案例】\n")
            f.write("=" * 60 + "\n\n")
            
            problem_cases = []
            
            # 找出有錯誤的案例
            for result in self.results:
                if 'error' in result:
                    problem_cases.append({
                        'id': result['query_id'],
                        'query': result['query'],
                        'issue': f"執行錯誤: {result['error']}"
                    })
            
            # 找出有重複的案例
            for result in self.results:
                if 'error' not in result and result['metrics']['has_duplicates']:
                    problem_cases.append({
                        'id': result['query_id'],
                        'query': result['query'],
                        'issue': "發現重複推薦"
                    })
            
            # 找出特徵匹配度低的案例
            for result in self.results:
                if 'error' not in result:
                    match_score = result['metrics']['feature_match_score']
                    if match_score is not None and match_score < 0.5:
                        problem_cases.append({
                            'id': result['query_id'],
                            'query': result['query'],
                            'issue': f"特徵匹配度低: {match_score}"
                        })
            
            if problem_cases:
                for i, case in enumerate(problem_cases, 1):
                    f.write(f"{i}. [{case['id']}] {case['query']}\n")
                    f.write(f"   問題: {case['issue']}\n\n")
            else:
                f.write("未發現問題案例\n\n")
            
            # 詳細結果
            f.write("=" * 60 + "\n")
            f.write("【詳細結果】\n")
            f.write("=" * 60 + "\n\n")
            
            for result in self.results:
                if 'error' in result:
                    f.write(f"[{result['query_id']}] {result['query']}\n")
                    f.write(f"錯誤: {result['error']}\n\n")
                    continue
                
                f.write(f"[{result['query_id']}] {result['query']}\n")
                f.write(f"- 類別: {result['category']}\n")
                f.write(f"- 回應時間: {result['total_time']} 秒\n")
                
                if result['metrics']['feature_match_score'] is not None:
                    f.write(f"- 特徵匹配度: {result['metrics']['feature_match_score']}")
                    if result['metrics']['feature_match_score'] >= 0.7:
                        f.write(" [良好]")
                    elif result['metrics']['feature_match_score'] >= 0.5:
                        f.write(" [尚可]")
                    else:
                        f.write(" [警告]")
                    f.write("\n")
                
                f.write(f"- 多樣性分數: {result['metrics']['diversity_score']}\n")
                f.write(f"- 重複: {'是' if result['metrics']['has_duplicates'] else '無'}")
                if result['metrics']['has_duplicates']:
                    f.write(" [警告]")
                f.write("\n")
                
                f.write("- 推薦歌曲:\n")
                for i, song in enumerate(result['recommendations'], 1):
                    f.write(f"  {i}. {song['track_name']} - {song['artists']}\n")
                    f.write(f"     類型: {song['track_genre']}, ")
                    f.write(f"能量: {song['energy']:.2f}, ")
                    f.write(f"節奏: {song['danceability']:.2f}\n")
                
                f.write("\n")
        
        print(f"\n評估報告已生成: {output_path}")
    
    def save_results(self, output_path='baseline_results.json'):
        """
        將所有結果儲存為 JSON
        
        參數:
            output_path: 輸出檔案路徑
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"測試結果已儲存: {output_path}")


def load_system():
    """
    載入向量資料庫和 LLM 模型
    
    返回:
        llm, vectordb
    """
    _ = load_dotenv(find_dotenv())
    
    # 初始化 Claude LLM
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        api_key=os.environ['CLAUDE_API_KEY']
    )
    
    # 初始化 Embedding
    embeddings = OpenAIEmbeddings(
        api_key=os.environ['OPENAI_API_KEY']
    )
    
    # 載入向量資料庫
    vectordb = Chroma(
        persist_directory="./spotify_chroma_db",
        embedding_function=embeddings
    )
    
    print(f"載入向量資料庫完成，共 {vectordb._collection.count()} 首歌")
    
    return llm, vectordb


if __name__ == "__main__":
    # 1. 載入系統
    print("載入系統...")
    llm, vectordb = load_system()
    
    # 2. 初始化評估器
    evaluator = MusicRecommendationEvaluator(llm, vectordb)
    
    # 3. 載入測試查詢
    test_queries = evaluator.load_test_queries("test_queries.json")
    
    # 4. 執行批次評估
    results = evaluator.run_batch_evaluation(test_queries)
    
    # 5. 儲存結果
    evaluator.save_results("baseline_results.json")
    
    # 6. 生成報告
    evaluator.generate_evaluation_report("evaluation_report.txt")
    
    print("\n評估完成！")
    print("- 結果已儲存至: baseline_results.json")
    print("- 報告已生成: evaluation_report.txt")