"""
RAG Manager

RAG 시스템의 통합 관리자
모든 RAG 기능을 하나의 인터페이스로 제공
"""

import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from .knowledge_store import KnowledgeStore
from .query_engine import QueryEngine
from .context_builder import ContextBuilder
from .vector_store import VectorStore
from .retriever import Retriever
from .rag_cache_manager import RAGCacheManager


class RAGManager:
    """RAG 시스템 통합 관리자"""
    
    def __init__(self, 
                 knowledge_base_path: Optional[str] = None,
                 cache_enabled: bool = True,
                 vector_store_type: str = "faiss"):
        """
        RAGManager 초기화
        
        Args:
            knowledge_base_path: 지식 베이스 경로
            cache_enabled: 캐싱 활성화 여부
            vector_store_type: 벡터 스토어 타입
        """
        self.logger = logging.getLogger(__name__)
        
        # 기본 경로 설정
        if knowledge_base_path is None:
            knowledge_base_path = "resources/knowledge_base"
        
        self.knowledge_base_path = Path(knowledge_base_path)
        self.cache_enabled = cache_enabled
        
        # RAG 컴포넌트 초기화
        try:
            self.vector_store = VectorStore()
            self.knowledge_store = KnowledgeStore(
                storage_path=str(self.knowledge_base_path)
            )
            self.query_engine = QueryEngine()
            self.retriever = Retriever(
                vector_store=self.vector_store
            )
            self.context_builder = ContextBuilder()
            
            if self.cache_enabled:
                self.cache_manager = RAGCacheManager()
            else:
                self.cache_manager = None
                
            self.is_initialized = True
            self.logger.info("RAGManager 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"RAGManager 초기화 실패: {e}")
            self.is_initialized = False
    
    def search(self, 
               query: str,
               collection: Optional[str] = None,
               top_k: int = 5,
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        통합 검색 인터페이스
        
        Args:
            query: 검색 쿼리
            collection: 검색할 컬렉션 (None이면 전체 검색)
            top_k: 반환할 결과 수
            filters: 추가 필터
            
        Returns:
            List[Dict]: 검색 결과
        """
        if not self.is_initialized:
            self.logger.error("RAGManager가 초기화되지 않았습니다")
            return []
        
        try:
            # 캐시 확인
            if self.cache_manager:
                cache_key = f"{query}_{collection}_{top_k}"
                cached_result = self.cache_manager.get(cache_key)
                if cached_result:
                    return cached_result
            
            # 검색 실행
            if collection:
                # QueryEngine을 사용한 검색 (knowledge_store 전달)
                results = self.query_engine.search(
                    query=query,
                    knowledge_store=self.knowledge_store,
                    knowledge_types=[collection] if collection else None,
                    filters=filters
                )
                # SearchResult 객체를 딕셔너리로 변환
                results = [self._search_result_to_dict(result) for result in results[:top_k]]
            else:
                # Retriever를 사용한 전체 검색
                results = self.retriever.retrieve(
                    query=query,
                    top_k=top_k,
                    filters=filters
                )
            
            # 캐시 저장
            if self.cache_manager and results:
                self.cache_manager.set(cache_key, results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"검색 오류: {e}")
            return []
    
    def _search_result_to_dict(self, search_result) -> Dict[str, Any]:
        """SearchResult 객체를 딕셔너리로 변환"""
        if hasattr(search_result, '__dict__'):
            return {
                'content': getattr(search_result, 'content', ''),
                'source': getattr(search_result, 'source', ''),
                'score': getattr(search_result, 'score', 0.0),
                'metadata': getattr(search_result, 'metadata', {}),
                'snippet': getattr(search_result, 'snippet', ''),
                'knowledge_type': getattr(search_result, 'knowledge_type', '')
            }
        return search_result if isinstance(search_result, dict) else {}
    
    def build_context(self, 
                     query: str,
                     search_results: Optional[List[Dict[str, Any]]] = None,
                     context_type: str = "general",
                     max_tokens: int = 2000) -> Dict[str, Any]:
        """
        컨텍스트 구성
        
        Args:
            query: 원본 쿼리
            search_results: 검색 결과 (None이면 자동 검색)
            context_type: 컨텍스트 타입
            max_tokens: 최대 토큰 수
            
        Returns:
            Dict: 구성된 컨텍스트
        """
        if not self.is_initialized:
            self.logger.error("RAGManager가 초기화되지 않았습니다")
            return {}
        
        try:
            # 검색 결과가 없으면 자동 검색
            if search_results is None:
                search_results = self.search(query)
            
            # 컨텍스트 구성
            context = self.context_builder.build_context(
                query=query,
                search_results=search_results,
                context_type=context_type,
                max_tokens=max_tokens
            )
            
            return context
            
        except Exception as e:
            self.logger.error(f"컨텍스트 구성 오류: {e}")
            return {}
    
    def search_and_build_context(self,
                                query: str,
                                collection: Optional[str] = None,
                                top_k: int = 5,
                                context_type: str = "general",
                                max_tokens: int = 2000) -> Dict[str, Any]:
        """
        검색과 컨텍스트 구성을 한번에 수행
        
        Args:
            query: 검색 쿼리
            collection: 검색할 컬렉션
            top_k: 반환할 결과 수
            context_type: 컨텍스트 타입
            max_tokens: 최대 토큰 수
            
        Returns:
            Dict: 검색 결과와 컨텍스트
        """
        if not self.is_initialized:
            return {
                'search_results': [],
                'context': {},
                'error': 'RAGManager가 초기화되지 않았습니다'
            }
        
        try:
            # 검색 실행
            search_results = self.search(
                query=query,
                collection=collection,
                top_k=top_k
            )
            
            # 컨텍스트 구성
            context = self.build_context(
                query=query,
                search_results=search_results,
                context_type=context_type,
                max_tokens=max_tokens
            )
            
            return {
                'search_results': search_results,
                'context': context,
                'query': query,
                'collection': collection
            }
            
        except Exception as e:
            self.logger.error(f"검색 및 컨텍스트 구성 오류: {e}")
            return {
                'search_results': [],
                'context': {},
                'error': str(e)
            }
    
    def get_knowledge_collections(self) -> List[str]:
        """사용 가능한 지식 컬렉션 목록 반환"""
        if not self.is_initialized:
            return []
        
        try:
            return self.knowledge_store.get_collections()
        except Exception as e:
            self.logger.error(f"컬렉션 목록 조회 오류: {e}")
            return []
    
    def get_collection_info(self, collection: str) -> Dict[str, Any]:
        """특정 컬렉션 정보 반환"""
        if not self.is_initialized:
            return {}
        
        try:
            return self.knowledge_store.get_collection_info(collection)
        except Exception as e:
            self.logger.error(f"컬렉션 정보 조회 오류: {e}")
            return {}
    
    def reload_knowledge_base(self) -> bool:
        """지식 베이스 재로드"""
        if not self.is_initialized:
            return False
        
        try:
            self.knowledge_store.reload()
            self.logger.info("지식 베이스 재로드 완료")
            return True
        except Exception as e:
            self.logger.error(f"지식 베이스 재로드 오류: {e}")
            return False
    
    def clear_cache(self) -> bool:
        """캐시 초기화"""
        if not self.cache_manager:
            return True
        
        try:
            self.cache_manager.clear()
            self.logger.info("RAG 캐시 초기화 완료")
            return True
        except Exception as e:
            self.logger.error(f"캐시 초기화 오류: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """RAG 시스템 통계 반환"""
        stats = {
            'initialized': self.is_initialized,
            'cache_enabled': self.cache_enabled,
            'knowledge_base_path': str(self.knowledge_base_path)
        }
        
        if self.is_initialized:
            try:
                stats['collections'] = self.get_knowledge_collections()
                stats['total_documents'] = sum(
                    self.get_collection_info(col).get('document_count', 0)
                    for col in stats['collections']
                )
                
                if self.cache_manager:
                    stats['cache_stats'] = self.cache_manager.get_stats()
                    
            except Exception as e:
                stats['error'] = str(e)
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """RAG 시스템 상태 확인"""
        health = {
            'status': 'healthy' if self.is_initialized else 'unhealthy',
            'components': {}
        }
        
        if self.is_initialized:
            # 각 컴포넌트 상태 확인
            components = [
                ('knowledge_store', self.knowledge_store),
                ('query_engine', self.query_engine),
                ('context_builder', self.context_builder),
                ('vector_store', self.vector_store),
                ('retriever', self.retriever)
            ]
            
            if self.cache_manager:
                components.append(('cache_manager', self.cache_manager))
            
            for name, component in components:
                try:
                    # 간단한 상태 확인 (hasattr로 기본 메서드 존재 확인)
                    if hasattr(component, 'health_check'):
                        health['components'][name] = component.health_check()
                    else:
                        health['components'][name] = {'status': 'available'}
                except Exception as e:
                    health['components'][name] = {
                        'status': 'error',
                        'error': str(e)
                    }
        
        return health 