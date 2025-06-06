# 파일명: services/rag/rag_service.py

import logging
from typing import Dict, Any, List, Optional

from .vector_store import VectorStore
from .knowledge_store import KnowledgeStore
from .retriever import Retriever
from .context_builder import ContextBuilder
# [UTIL-REQ] global_cache.py의 cached 데코레이터가 필요합니다.
from utils.global_cache import cached

logger = logging.getLogger(__name__)

class RAGService:
    """RAG 관련 기능을 통합 제공하는 Facade 클래스"""

    def __init__(self, storage_path: str = "output_data/rag_storage"):
        self._vector_store = VectorStore(storage_path=storage_path)
        self._knowledge_store = KnowledgeStore(self._vector_store)
        self._retriever = Retriever(self._vector_store)
        self._context_builder = ContextBuilder()
        self._initialized = False
        logger.info("RAG 서비스가 초기화되었습니다.")

    def initialize_knowledge_base(self, force_reingest: bool = False):
        """지식 베이스를 초기화(필요 시 문서 로드)합니다."""
        if self._initialized and not force_reingest:
            return
            
        # [TODO] 어떤 컬렉션을 로드할지 결정하는 로직 필요
        # 예시로 'statistical_concepts'와 'code_templates'를 로드
        self._knowledge_store.ingest_directory('statistical_concepts')
        self._knowledge_store.ingest_directory('code_templates/python')
        self._initialized = True
        logger.info("지식 베이스 초기화 및 문서 로드가 완료되었습니다.")
    
    @cached(ttl_minutes=10, key_prefix="rag_search")
    def search_and_build_context(
        self,
        query: str,
        collection: Optional[str] = None, # collection 필터링은 Retriever에서 구현 필요
        top_k: int = 3,
        max_context_length: int = 3000
    ) -> str:
        """검색과 컨텍스트 구성을 한 번에 수행하여 프롬프트에 바로 사용할 텍스트를 반환합니다."""
        if not self._initialized:
            self.initialize_knowledge_base()

        # 1. 문서 검색
        search_results = self._retriever.retrieve(query, top_k)
        
        # 2. 컨텍스트 구성
        context_text = self._context_builder.build_context(search_results, max_length=max_context_length)
        
        return context_text