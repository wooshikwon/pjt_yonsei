"""
RAG 시스템의 쿼리 엔진 모듈
검색 및 랭킹 로직을 담당하며, 다양한 유형의 지식에 대한 검색을 지원합니다.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import json
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime, timedelta

from utils.error_handler import ErrorHandler
from utils.global_cache import GlobalCache

@dataclass
class SearchResult:
    """검색 결과를 나타내는 데이터 클래스"""
    content: str
    source: str
    score: float
    metadata: Dict[str, Any]
    snippet: str
    knowledge_type: str
    
class QueryEngine:
    """RAG 시스템의 쿼리 엔진"""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 cache_ttl: int = 3600,
                 max_results: int = 10):
        """
        Args:
            embedding_model: 임베딩에 사용할 모델명
            cache_ttl: 캐시 유지 시간 (초)
            max_results: 반환할 최대 결과 수
        """
        self.logger = logging.getLogger(__name__)
        self.error_handler = ErrorHandler()
        self.cache = GlobalCache()
        self.cache_ttl = cache_ttl
        self.max_results = max_results
        
        # 임베딩 모델 초기화
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            self.logger.info(f"임베딩 모델 로드 완료: {embedding_model}")
        except Exception as e:
            self.logger.error(f"임베딩 모델 로드 실패: {e}")
            self.embedding_model = None
        
        # 검색 가중치 설정
        self.search_weights = {
            'semantic_similarity': 0.6,
            'keyword_match': 0.3,
            'metadata_relevance': 0.1
        }
        
        # 지식 유형별 가중치
        self.knowledge_type_weights = {
            'statistical_concepts': 1.0,
            'business_domains': 0.9,
            'code_templates': 0.8,
            'workflow_guidelines': 0.7
        }
    
    def search(self, 
               query: str,
               knowledge_store: Any,
               knowledge_types: Optional[List[str]] = None,
               filters: Optional[Dict[str, Any]] = None,
               include_metadata: bool = True) -> List[SearchResult]:
        """
        지식 저장소에서 쿼리에 맞는 정보를 검색
        
        Args:
            query: 검색 쿼리
            knowledge_store: 지식 저장소 인스턴스
            knowledge_types: 검색할 지식 유형 리스트
            filters: 추가 필터 조건
            include_metadata: 메타데이터 포함 여부
            
        Returns:
            검색 결과 리스트
        """
        try:
            # 캐시 키 생성
            cache_key = self._generate_cache_key(query, knowledge_types, filters)
            cached_result = self.cache.get(cache_key)
            
            if cached_result:
                self.logger.debug(f"캐시에서 검색 결과 반환: {query[:50]}...")
                return cached_result
            
            # 쿼리 전처리
            processed_query = self._preprocess_query(query)
            
            # 하이브리드 검색 수행
            search_results = self._hybrid_search(
                processed_query, knowledge_store, knowledge_types, filters
            )
            
            # 결과 랭킹 및 정제
            ranked_results = self._rank_and_filter_results(search_results, processed_query)
            
            # 스니펫 생성
            final_results = self._generate_snippets(ranked_results, processed_query)
            
            # 메타데이터 추가
            if include_metadata:
                final_results = self._enrich_with_metadata(final_results)
            
            # 캐시에 저장
            self.cache.set(cache_key, final_results, ttl=self.cache_ttl)
            
            self.logger.info(f"검색 완료: {len(final_results)}개 결과 반환")
            return final_results[:self.max_results]
            
        except Exception as e:
            self.logger.error(f"검색 오류: {e}")
            return self.error_handler.handle_error(e, default_return=[])
    
    def multi_query_search(self,
                          queries: List[str],
                          knowledge_store: Any,
                          fusion_method: str = "rrf") -> List[SearchResult]:
        """
        여러 쿼리를 사용한 다중 검색 및 결과 융합
        
        Args:
            queries: 쿼리 리스트
            knowledge_store: 지식 저장소
            fusion_method: 융합 방법 ("rrf", "weighted", "concat")
            
        Returns:
            융합된 검색 결과
        """
        try:
            all_results = []
            
            for query in queries:
                results = self.search(query, knowledge_store)
                all_results.append(results)
            
            # 결과 융합
            if fusion_method == "rrf":
                fused_results = self._reciprocal_rank_fusion(all_results)
            elif fusion_method == "weighted":
                fused_results = self._weighted_fusion(all_results, queries)
            else:  # concat
                fused_results = self._concatenate_fusion(all_results)
            
            return fused_results[:self.max_results]
            
        except Exception as e:
            self.logger.error(f"다중 쿼리 검색 오류: {e}")
            return []
    
    def semantic_search(self,
                       query: str,
                       knowledge_store: Any,
                       similarity_threshold: float = 0.5) -> List[SearchResult]:
        """
        순수 의미론적 검색
        
        Args:
            query: 검색 쿼리
            knowledge_store: 지식 저장소
            similarity_threshold: 유사도 임계값
            
        Returns:
            의미론적 검색 결과
        """
        try:
            if not self.embedding_model:
                raise ValueError("임베딩 모델이 초기화되지 않았습니다")
            
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_model.encode([query])
            
            # 지식 저장소에서 임베딩 검색
            documents = knowledge_store.get_all_documents()
            doc_embeddings = knowledge_store.get_embeddings()
            
            if not doc_embeddings:
                return []
            
            # 코사인 유사도 계산
            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
            
            # 임계값 이상의 결과만 선택
            valid_indices = np.where(similarities >= similarity_threshold)[0]
            
            results = []
            for idx in valid_indices:
                doc = documents[idx]
                result = SearchResult(
                    content=doc.get('content', ''),
                    source=doc.get('source', ''),
                    score=float(similarities[idx]),
                    metadata=doc.get('metadata', {}),
                    snippet='',
                    knowledge_type=doc.get('type', 'unknown')
                )
                results.append(result)
            
            # 점수순 정렬
            results.sort(key=lambda x: x.score, reverse=True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"의미론적 검색 오류: {e}")
            return []
    
    def keyword_search(self,
                      query: str,
                      knowledge_store: Any,
                      match_type: str = "fuzzy") -> List[SearchResult]:
        """
        키워드 기반 검색
        
        Args:
            query: 검색 쿼리
            knowledge_store: 지식 저장소
            match_type: 매칭 타입 ("exact", "fuzzy", "regex")
            
        Returns:
            키워드 검색 결과
        """
        try:
            documents = knowledge_store.get_all_documents()
            keywords = self._extract_keywords(query)
            
            results = []
            for doc in documents:
                content = doc.get('content', '').lower()
                score = 0.0
                
                if match_type == "exact":
                    score = self._exact_keyword_match(keywords, content)
                elif match_type == "fuzzy":
                    score = self._fuzzy_keyword_match(keywords, content)
                elif match_type == "regex":
                    score = self._regex_keyword_match(keywords, content)
                
                if score > 0:
                    result = SearchResult(
                        content=doc.get('content', ''),
                        source=doc.get('source', ''),
                        score=score,
                        metadata=doc.get('metadata', {}),
                        snippet='',
                        knowledge_type=doc.get('type', 'unknown')
                    )
                    results.append(result)
            
            # 점수순 정렬
            results.sort(key=lambda x: x.score, reverse=True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"키워드 검색 오류: {e}")
            return []
    
    def _hybrid_search(self,
                      query: str,
                      knowledge_store: Any,
                      knowledge_types: Optional[List[str]],
                      filters: Optional[Dict[str, Any]]) -> List[SearchResult]:
        """하이브리드 검색 수행"""
        # 의미론적 검색
        semantic_results = self.semantic_search(query, knowledge_store)
        
        # 키워드 검색
        keyword_results = self.keyword_search(query, knowledge_store, "fuzzy")
        
        # 결과 병합 및 점수 조합
        combined_results = self._combine_search_results(
            semantic_results, keyword_results
        )
        
        # 지식 유형 필터링
        if knowledge_types:
            combined_results = [
                r for r in combined_results 
                if r.knowledge_type in knowledge_types
            ]
        
        # 추가 필터 적용
        if filters:
            combined_results = self._apply_filters(combined_results, filters)
        
        return combined_results
    
    def _combine_search_results(self,
                               semantic_results: List[SearchResult],
                               keyword_results: List[SearchResult]) -> List[SearchResult]:
        """검색 결과들을 병합하고 점수를 조합"""
        result_map = {}
        
        # 의미론적 결과 추가
        for result in semantic_results:
            key = f"{result.source}:{hash(result.content[:100])}"
            if key not in result_map:
                result_map[key] = result
                result_map[key].score *= self.search_weights['semantic_similarity']
            else:
                result_map[key].score += result.score * self.search_weights['semantic_similarity']
        
        # 키워드 결과 추가
        for result in keyword_results:
            key = f"{result.source}:{hash(result.content[:100])}"
            if key not in result_map:
                result_map[key] = result
                result_map[key].score *= self.search_weights['keyword_match']
            else:
                result_map[key].score += result.score * self.search_weights['keyword_match']
        
        # 지식 유형별 가중치 적용
        for result in result_map.values():
            type_weight = self.knowledge_type_weights.get(result.knowledge_type, 1.0)
            result.score *= type_weight
        
        return list(result_map.values())
    
    def _rank_and_filter_results(self,
                                results: List[SearchResult],
                                query: str) -> List[SearchResult]:
        """결과 랭킹 및 필터링"""
        # 점수순 정렬
        results.sort(key=lambda x: x.score, reverse=True)
        
        # 중복 제거 (내용 기반)
        unique_results = []
        seen_content = set()
        
        for result in results:
            content_hash = hash(result.content[:200])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        # 최소 점수 임계값 적용
        min_score = 0.1
        filtered_results = [r for r in unique_results if r.score >= min_score]
        
        return filtered_results
    
    def _generate_snippets(self,
                          results: List[SearchResult],
                          query: str) -> List[SearchResult]:
        """검색 결과에 대한 스니펫 생성"""
        query_terms = self._extract_keywords(query)
        
        for result in results:
            result.snippet = self._create_snippet(result.content, query_terms)
        
        return results
    
    def _create_snippet(self, content: str, query_terms: List[str], 
                       max_length: int = 200) -> str:
        """단일 스니펫 생성"""
        try:
            content_lower = content.lower()
            
            # 쿼리 용어가 포함된 문장 찾기
            sentences = content.split('.')
            best_sentence = ""
            max_matches = 0
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                matches = sum(1 for term in query_terms if term.lower() in sentence_lower)
                
                if matches > max_matches:
                    max_matches = matches
                    best_sentence = sentence.strip()
            
            # 적절한 길이로 자르기
            if len(best_sentence) > max_length:
                # 단어 경계에서 자르기
                words = best_sentence.split()
                snippet = ""
                for word in words:
                    if len(snippet + word) < max_length - 3:
                        snippet += word + " "
                    else:
                        break
                best_sentence = snippet.strip() + "..."
            
            return best_sentence if best_sentence else content[:max_length] + "..."
            
        except Exception:
            return content[:max_length] + "..."
    
    def _enrich_with_metadata(self, results: List[SearchResult]) -> List[SearchResult]:
        """메타데이터로 결과 보강"""
        for result in results:
            # 관련성 점수 추가
            result.metadata['relevance_score'] = result.score
            
            # 검색 시간 추가
            result.metadata['search_timestamp'] = datetime.now().isoformat()
            
            # 콘텐츠 길이 추가
            result.metadata['content_length'] = len(result.content)
            
            # 키워드 밀도 계산 (간단한 버전)
            word_count = len(result.content.split())
            result.metadata['keyword_density'] = word_count / max(len(result.content), 1)
        
        return results
    
    def _preprocess_query(self, query: str) -> str:
        """쿼리 전처리"""
        # 소문자 변환
        query = query.lower().strip()
        
        # 특수 문자 정규화
        query = re.sub(r'[^\w\s가-힣]', ' ', query)
        
        # 중복 공백 제거
        query = re.sub(r'\s+', ' ', query)
        
        return query
    
    def _extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 키워드 추출"""
        # 간단한 키워드 추출 (실제로는 더 정교한 NLP 기법 사용 가능)
        words = text.lower().split()
        
        # 불용어 제거 (간단한 버전)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                    'of', 'with', 'by', '은', '는', '이', '가', '을', '를', '에', '의', '와', '과'}
        
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        
        return keywords
    
    def _exact_keyword_match(self, keywords: List[str], content: str) -> float:
        """정확한 키워드 매칭 점수 계산"""
        matches = sum(1 for keyword in keywords if keyword in content)
        return matches / len(keywords) if keywords else 0.0
    
    def _fuzzy_keyword_match(self, keywords: List[str], content: str) -> float:
        """퍼지 키워드 매칭 점수 계산"""
        total_score = 0.0
        
        for keyword in keywords:
            # 부분 문자열 매칭
            if keyword in content:
                total_score += 1.0
            else:
                # 편집 거리 기반 퍼지 매칭 (간단한 버전)
                words = content.split()
                for word in words:
                    if len(word) > 2:
                        similarity = self._string_similarity(keyword, word)
                        if similarity > 0.8:
                            total_score += similarity
                            break
        
        return total_score / len(keywords) if keywords else 0.0
    
    def _regex_keyword_match(self, keywords: List[str], content: str) -> float:
        """정규식 기반 키워드 매칭"""
        matches = 0
        
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, content, re.IGNORECASE):
                matches += 1
        
        return matches / len(keywords) if keywords else 0.0
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """두 문자열 간의 유사도 계산 (간단한 Jaccard 유사도)"""
        set1 = set(s1.lower())
        set2 = set(s2.lower())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _apply_filters(self, results: List[SearchResult], 
                      filters: Dict[str, Any]) -> List[SearchResult]:
        """결과에 필터 적용"""
        filtered_results = results
        
        # 날짜 필터
        if 'date_range' in filters:
            date_range = filters['date_range']
            filtered_results = [
                r for r in filtered_results
                if self._check_date_filter(r, date_range)
            ]
        
        # 점수 필터
        if 'min_score' in filters:
            min_score = filters['min_score']
            filtered_results = [
                r for r in filtered_results
                if r.score >= min_score
            ]
        
        # 소스 필터
        if 'sources' in filters:
            allowed_sources = filters['sources']
            filtered_results = [
                r for r in filtered_results
                if r.source in allowed_sources
            ]
        
        return filtered_results
    
    def _check_date_filter(self, result: SearchResult, date_range: Dict[str, str]) -> bool:
        """날짜 필터 확인"""
        try:
            if 'last_modified' in result.metadata:
                last_modified = datetime.fromisoformat(result.metadata['last_modified'])
                
                if 'start_date' in date_range:
                    start_date = datetime.fromisoformat(date_range['start_date'])
                    if last_modified < start_date:
                        return False
                
                if 'end_date' in date_range:
                    end_date = datetime.fromisoformat(date_range['end_date'])
                    if last_modified > end_date:
                        return False
            
            return True
            
        except Exception:
            return True  # 날짜 파싱 실패 시 포함
    
    def _reciprocal_rank_fusion(self, result_lists: List[List[SearchResult]]) -> List[SearchResult]:
        """Reciprocal Rank Fusion 알고리즘으로 결과 융합"""
        rrf_scores = {}
        k = 60  # RRF 상수
        
        for results in result_lists:
            for rank, result in enumerate(results, 1):
                key = f"{result.source}:{hash(result.content[:100])}"
                if key not in rrf_scores:
                    rrf_scores[key] = {'result': result, 'score': 0}
                rrf_scores[key]['score'] += 1 / (k + rank)
        
        # 점수순 정렬
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        # SearchResult 객체들을 반환하되 점수 업데이트
        final_results = []
        for item in sorted_results:
            result = item['result']
            result.score = item['score']
            final_results.append(result)
        
        return final_results
    
    def _weighted_fusion(self, result_lists: List[List[SearchResult]], 
                        queries: List[str]) -> List[SearchResult]:
        """가중치 기반 융합"""
        # 쿼리 중요도에 따른 가중치 (간단한 버전)
        weights = [1.0 / len(queries)] * len(queries)
        
        weighted_scores = {}
        
        for results, weight in zip(result_lists, weights):
            for result in results:
                key = f"{result.source}:{hash(result.content[:100])}"
                if key not in weighted_scores:
                    weighted_scores[key] = {'result': result, 'score': 0}
                weighted_scores[key]['score'] += result.score * weight
        
        # 점수순 정렬
        sorted_results = sorted(
            weighted_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        # SearchResult 객체들을 반환하되 점수 업데이트
        final_results = []
        for item in sorted_results:
            result = item['result']
            result.score = item['score']
            final_results.append(result)
        
        return final_results
    
    def _concatenate_fusion(self, result_lists: List[List[SearchResult]]) -> List[SearchResult]:
        """단순 연결 융합"""
        all_results = []
        seen_keys = set()
        
        for results in result_lists:
            for result in results:
                key = f"{result.source}:{hash(result.content[:100])}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_results.append(result)
        
        # 원래 점수 기준으로 정렬
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        return all_results
    
    def _generate_cache_key(self, query: str, knowledge_types: Optional[List[str]], 
                           filters: Optional[Dict[str, Any]]) -> str:
        """캐시 키 생성"""
        key_parts = [query]
        
        if knowledge_types:
            key_parts.append(",".join(sorted(knowledge_types)))
        
        if filters:
            # 필터를 정렬 가능한 문자열로 변환
            filter_str = json.dumps(filters, sort_keys=True)
            key_parts.append(filter_str)
        
        return "query_engine:" + ":".join(key_parts)
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """검색 통계 정보 반환"""
        return {
            'embedding_model': str(self.embedding_model),
            'cache_ttl': self.cache_ttl,
            'max_results': self.max_results,
            'search_weights': self.search_weights,
            'knowledge_type_weights': self.knowledge_type_weights
        } 