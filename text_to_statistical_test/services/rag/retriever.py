# 파일명: services/rag/retriever.py

import logging
from typing import Dict, List, Any

from .vector_store import VectorStore
# [UTIL-REQ] helpers.py에 extract_keywords, calculate_text_similarity 함수가 필요합니다.
from utils.helpers import extract_keywords, calculate_text_similarity

logger = logging.getLogger(__name__)

class Retriever:
    """VectorStore를 사용하여 다양한 전략으로 문서를 검색하고 순위를 재조정합니다."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5, strategy: str = "hybrid") -> List[Dict[str, Any]]:
        """주어진 전략에 따라 문서를 검색합니다."""
        if strategy == "vector":
            return self.vector_store.search(query, top_k)
        elif strategy == "hybrid":
            return self._hybrid_search(query, top_k)
        else:
            raise ValueError(f"지원하지 않는 검색 전략: {strategy}")

    def _hybrid_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """벡터 검색과 키워드 검색을 결합한 하이브리드 검색을 수행합니다."""
        # 더 많은 후보군을 확보하기 위해 top_k의 3배수만큼 검색
        vector_results = self.vector_store.search(query, top_k * 3)
        
        query_keywords = set(extract_keywords(query, max_words=5))
        if not query_keywords:
            return vector_results[:top_k]

        # 키워드 점수와 텍스트 유사도(재정렬) 점수 추가
        for res in vector_results:
            text_keywords = set(extract_keywords(res['text']))
            keyword_score = len(query_keywords.intersection(text_keywords)) / len(query_keywords)
            rerank_score = calculate_text_similarity(query, res['text'])
            
            # 최종 점수 = 벡터 점수 * 0.5 + 키워드 점수 * 0.2 + 재정렬 점수 * 0.3
            res['final_score'] = res['score'] * 0.5 + keyword_score * 0.2 + rerank_score * 0.3
        
        # 최종 점수로 정렬
        vector_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return vector_results[:top_k]