# src/rag_retriever.py

import logging
import os
from typing import List, Dict, Any, Optional

# Vector DB 및 임베딩 관련 라이브러리 임포트 (requirements.txt에 명시 필요)
# 예시:
# from chromadb import PersistentClient
# from chromadb.utils import embedding_functions
# 또는 Langchain 사용 시:
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings # 또는 OpenAIEmbeddings 등

try:
    from chromadb import PersistentClient
    from sentence_transformers import SentenceTransformer
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

logger = logging.getLogger(__name__)

class RAGRetriever:
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.rag_config = settings.get('rag', {})
        self.vector_db_path = self.rag_config.get('vector_db_path')
        self.embedding_model_name = self.rag_config.get('embedding_model_name')
        self.top_k = self.rag_config.get('rag_top_k', 3)
        
        self.db_client = None # 예: ChromaDB 클라이언트
        self.collection = None # 예: ChromaDB 컬렉션
        self.embedding_function = None # 예: SentenceTransformer 임베딩 함수

        if not self.vector_db_path or not self.embedding_model_name:
            logger.error("Vector DB path or embedding model name not configured. RAG will not function.")
            self.is_initialized = False
            return

        self._initialize_retriever()
        logger.info(f"RAGRetriever initialized. DB path: {self.vector_db_path}, Embedding: {self.embedding_model_name}")

    def _initialize_retriever(self):
        """
        Vector DB 클라이언트 및 임베딩 함수를 초기화합니다.
        PoC에서는 `scripts/build_vector_db.py`에서 생성한 DB를 로드합니다.
        """
        try:
            if not os.path.exists(self.vector_db_path):
                logger.warning(f"Vector DB directory not found at {self.vector_db_path}. RAG retrieval will return empty.")
                self.is_initialized = False
                return
            if CHROMA_AVAILABLE:
                self.embedding_function = SentenceTransformer(self.embedding_model_name)
                self.db_client = PersistentClient(path=self.vector_db_path)
                collection_name = os.path.basename(self.vector_db_path) or "jeonse_fraud_collection"
                self.collection = self.db_client.get_or_create_collection(name=collection_name)
                logger.info(f"Successfully initialized ChromaDB client and got collection '{collection_name}'.")
                self.is_initialized = True
            else:
                logger.warning("ChromaDB or SentenceTransformer not available. Using mock RAG retrieval.")
                self.is_initialized = True # fallback to mock
        except Exception as e:
            logger.error(f"Failed to initialize RAG retriever: {e}", exc_info=True)
            self.is_initialized = False

    def retrieve_documents(self, query: str, custom_top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.is_initialized:
            logger.warning("RAG retriever not initialized. Returning empty list.")
            return []

        current_top_k = custom_top_k if custom_top_k is not None else self.top_k
        logger.info(f"Retrieving top {current_top_k} documents from RAG for query (first 50 chars): '{query[:50]}...'")
        
        try:
            if CHROMA_AVAILABLE and self.collection and self.embedding_function:
                query_embedding = self.embedding_function.encode([query])[0]
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=current_top_k,
                    include=["documents", "metadatas", "distances"]
                )
                retrieved_docs = []
                if results.get('documents') and results.get('metadatas'):
                    for i, doc_content in enumerate(results['documents'][0]):
                        retrieved_docs.append({
                            "content": doc_content,
                            "metadata": results['metadatas'][0][i],
                            "score": 1 - results['distances'][0][i] if results.get('distances') else None
                        })
                logger.info(f"Retrieved {len(retrieved_docs)} documents from Vector DB.")
                return retrieved_docs
            # fallback to mock
            mock_retrieved_docs = [
                {"content": f"모의 RAG 문서 1: '{query}' 관련 판례입니다. [판례번호 2023가단12345] 전세금 반환 소송에서 임차인이 승소한 사례...", "metadata": {"source": "legal_cases/sim_case_01.txt", "type": "판례"}, "score": 0.85},
                {"content": f"모의 RAG 문서 2: '{query}' 관련 법령입니다. 주택임대차보호법 제3조의2 (보증금의 회수)에 따르면...", "metadata": {"source": "statutes/housing_protection_act.txt", "type": "법령"}, "score": 0.78},
                {"content": f"모의 RAG 문서 3: '{query}' 관련 사기 예방 가이드입니다. 계약 전 등기부등본 '을구'의 근저당 설정을 반드시 확인하세요...", "metadata": {"source": "prevention_guides/contract_checklist.txt", "type": "가이드"}, "score": 0.75},
            ]
            logger.info(f"Mock RAG: Retrieved {len(mock_retrieved_docs)} documents for query '{query}'.")
            return mock_retrieved_docs[:current_top_k]

        except Exception as e:
            logger.error(f"Error during RAG document retrieval: {e}", exc_info=True)
            return []