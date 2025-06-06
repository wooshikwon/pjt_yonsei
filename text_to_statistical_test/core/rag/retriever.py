"""
Retriever

RAG 시스템을 위한 문서 검색 및 순위화 컴포넌트
다양한 검색 전략과 재순위화 기법 제공
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import re

from .vector_store import VectorStore
from utils.helpers import calculate_text_similarity, extract_keywords, clean_text
from utils.global_cache import get_global_cache
from utils.error_handler import ErrorHandler, RAGError

logger = logging.getLogger(__name__)

class Retriever:
    """
    다양한 검색 전략을 지원하는 문서 검색기
    """
    
    def __init__(self, 
                 vector_store: VectorStore,
                 default_k: int = 5,
                 rerank: bool = True,
                 diversity_threshold: float = 0.7):
        """
        검색기 초기화
        
        Args:
            vector_store: 벡터 저장소
            default_k: 기본 검색 결과 수
            rerank: 재순위화 사용 여부
            diversity_threshold: 다양성 임계값
        """
        self.vector_store = vector_store
        self.default_k = default_k
        self.rerank = rerank
        self.diversity_threshold = diversity_threshold
        
        self.cache = get_global_cache()
        self.error_handler = ErrorHandler()
        
        logger.info("검색기 초기화 완료")
    
    def retrieve(self, 
                query: str,
                k: Optional[int] = None,
                search_type: str = "hybrid",
                filter_metadata: Optional[Dict[str, Any]] = None,
                context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        문서 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            search_type: 검색 유형 ("vector", "keyword", "hybrid")
            filter_metadata: 메타데이터 필터
            context: 추가 컨텍스트 정보
            
        Returns:
            List[Dict[str, Any]]: 검색 결과
        """
        try:
            k = k or self.default_k
            
            # 캐시 확인
            cache_key = f"retrieve_{hash((query, k, search_type, str(filter_metadata)))}"
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                logger.debug("캐시된 검색 결과 반환")
                return cached_result
            
            logger.info(f"문서 검색 시작: '{query[:50]}...' (유형: {search_type})")
            
            # 검색 유형별 처리
            if search_type == "vector":
                results = self._vector_search(query, k, filter_metadata)
            elif search_type == "keyword":
                results = self._keyword_search(query, k, filter_metadata)
            elif search_type == "hybrid":
                results = self._hybrid_search(query, k, filter_metadata)
            else:
                raise RAGError(f"지원하지 않는 검색 유형: {search_type}")
            
            # 컨텍스트 기반 후처리
            if context:
                results = self._apply_context_filtering(results, context)
            
            # 재순위화
            if self.rerank and len(results) > 1:
                results = self._rerank_results(query, results)
            
            # 다양성 확보
            results = self._ensure_diversity(results)
            
            # 최종 결과 제한
            results = results[:k]
            
            # 캐시 저장
            self.cache.set(cache_key, results, ttl=300)  # 5분 캐시
            
            logger.info(f"검색 완료: {len(results)}개 결과 반환")
            return results
            
        except Exception as e:
            logger.error(f"문서 검색 실패: {str(e)}")
            raise RAGError(f"문서 검색 실패: {str(e)}")
    
    def retrieve_similar_documents(self, 
                                 doc_id: str, 
                                 k: int = 5) -> List[Dict[str, Any]]:
        """
        특정 문서와 유사한 문서들 검색
        
        Args:
            doc_id: 기준 문서 ID
            k: 반환할 결과 수
            
        Returns:
            List[Dict[str, Any]]: 유사 문서 리스트
        """
        try:
            # 기준 문서 조회
            base_doc = self.vector_store.get_document(doc_id)
            if not base_doc:
                raise RAGError(f"문서를 찾을 수 없습니다: {doc_id}")
            
            # 기준 문서의 텍스트로 검색
            results = self.vector_store.search(base_doc['text'], k + 1)  # +1은 자기 자신 제외용
            
            # 자기 자신 제외
            filtered_results = [r for r in results if r['doc_id'] != doc_id]
            
            return filtered_results[:k]
            
        except Exception as e:
            logger.error(f"유사 문서 검색 실패: {str(e)}")
            raise RAGError(f"유사 문서 검색 실패: {str(e)}")
    
    def retrieve_by_keywords(self, 
                           keywords: List[str], 
                           k: int = 10,
                           match_mode: str = "any") -> List[Dict[str, Any]]:
        """
        키워드 기반 문서 검색
        
        Args:
            keywords: 검색 키워드 리스트
            k: 반환할 결과 수
            match_mode: 매칭 모드 ("any", "all")
            
        Returns:
            List[Dict[str, Any]]: 검색 결과
        """
        try:
            all_docs = self.vector_store.list_documents()
            results = []
            
            for doc_info in all_docs:
                doc_id = doc_info['doc_id']
                doc = self.vector_store.get_document(doc_id)
                if not doc:
                    continue
                
                text = doc['text'].lower()
                matched_keywords = []
                
                for keyword in keywords:
                    if keyword.lower() in text:
                        matched_keywords.append(keyword)
                
                # 매칭 모드에 따른 필터링
                if match_mode == "all" and len(matched_keywords) != len(keywords):
                    continue
                elif match_mode == "any" and len(matched_keywords) == 0:
                    continue
                
                # 매칭 점수 계산
                match_score = len(matched_keywords) / len(keywords)
                
                result = {
                    'doc_id': doc_id,
                    'text': doc['text'],
                    'metadata': doc['metadata'],
                    'score': match_score,
                    'matched_keywords': matched_keywords
                }
                results.append(result)
            
            # 점수 기준 정렬
            results.sort(key=lambda x: x['score'], reverse=True)
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"키워드 검색 실패: {str(e)}")
            raise RAGError(f"키워드 검색 실패: {str(e)}")
    
    def _vector_search(self, 
                      query: str, 
                      k: int, 
                      filter_metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """벡터 유사도 검색"""
        return self.vector_store.search(query, k, filter_metadata)
    
    def _keyword_search(self, 
                       query: str, 
                       k: int, 
                       filter_metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """키워드 기반 검색"""
        # 쿼리에서 키워드 추출
        keywords = extract_keywords(query)
        if not keywords:
            keywords = query.split()
        
        return self.retrieve_by_keywords(keywords, k, "any")
    
    def _hybrid_search(self, 
                      query: str, 
                      k: int, 
                      filter_metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """하이브리드 검색 (벡터 + 키워드)"""
        # 벡터 검색 결과
        vector_results = self._vector_search(query, k * 2, filter_metadata)
        
        # 키워드 검색 결과
        keyword_results = self._keyword_search(query, k * 2, filter_metadata)
        
        # 결과 결합 및 점수 조정
        combined_results = {}
        
        # 벡터 검색 결과 처리 (가중치 0.7)
        for result in vector_results:
            doc_id = result['doc_id']
            combined_results[doc_id] = {
                **result,
                'vector_score': result['score'] * 0.7,
                'keyword_score': 0.0,
                'combined_score': result['score'] * 0.7
            }
        
        # 키워드 검색 결과 처리 (가중치 0.3)
        for result in keyword_results:
            doc_id = result['doc_id']
            keyword_score = result['score'] * 0.3
            
            if doc_id in combined_results:
                # 기존 결과에 키워드 점수 추가
                combined_results[doc_id]['keyword_score'] = keyword_score
                combined_results[doc_id]['combined_score'] += keyword_score
            else:
                # 새로운 결과 추가
                combined_results[doc_id] = {
                    **result,
                    'vector_score': 0.0,
                    'keyword_score': keyword_score,
                    'combined_score': keyword_score,
                    'score': keyword_score  # 기본 score 필드
                }
        
        # 점수 기준 정렬
        final_results = list(combined_results.values())
        final_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return final_results[:k]
    
    def _apply_context_filtering(self, 
                               results: List[Dict[str, Any]], 
                               context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """컨텍스트 기반 결과 필터링"""
        # 분석 유형별 필터링
        analysis_type = context.get('analysis_type')
        if analysis_type:
            filtered_results = []
            for result in results:
                metadata = result.get('metadata', {})
                doc_analysis_types = metadata.get('analysis_types', [])
                
                if not doc_analysis_types or analysis_type in doc_analysis_types:
                    filtered_results.append(result)
            
            if filtered_results:
                results = filtered_results
        
        # 데이터 유형별 필터링
        data_type = context.get('data_type')
        if data_type:
            filtered_results = []
            for result in results:
                metadata = result.get('metadata', {})
                doc_data_types = metadata.get('data_types', [])
                
                if not doc_data_types or data_type in doc_data_types:
                    filtered_results.append(result)
            
            if filtered_results:
                results = filtered_results
        
        return results
    
    def _rerank_results(self, 
                       query: str, 
                       results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """검색 결과 재순위화"""
        try:
            # 쿼리와 문서의 의미적 유사도 기반 재순위화
            for result in results:
                text = result['text']
                
                # 텍스트 유사도 계산
                similarity = calculate_text_similarity(query, text)
                
                # 기존 점수와 결합
                original_score = result.get('score', 0)
                new_score = (original_score * 0.6) + (similarity * 0.4)
                
                result['rerank_score'] = new_score
                result['text_similarity'] = similarity
            
            # 새로운 점수로 정렬
            results.sort(key=lambda x: x.get('rerank_score', x.get('score', 0)), reverse=True)
            
            return results
            
        except Exception as e:
            logger.warning(f"재순위화 실패, 원본 결과 반환: {str(e)}")
            return results
    
    def _ensure_diversity(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """결과 다양성 확보"""
        if len(results) <= 1:
            return results
        
        diverse_results = [results[0]]  # 첫 번째 결과는 항상 포함
        
        for result in results[1:]:
            is_diverse = True
            
            for existing_result in diverse_results:
                # 텍스트 유사도 계산
                similarity = calculate_text_similarity(
                    result['text'], existing_result['text']
                )
                
                if similarity > self.diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_results.append(result)
        
        return diverse_results
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """검색 통계 정보"""
        vector_stats = self.vector_store.get_stats()
        
        return {
            'vector_store_stats': vector_stats,
            'default_k': self.default_k,
            'rerank_enabled': self.rerank,
            'diversity_threshold': self.diversity_threshold,
            'cache_stats': self.cache.get_stats()
        } 