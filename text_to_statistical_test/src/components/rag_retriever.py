import os
import shutil
from pathlib import Path
from typing import List
import logging

# 통합된 경고 설정 사용
from src.utils.warnings_config import suppress_stdout

# LlamaIndex 관련 임포트
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# FAISS 라이브러리 임포트
import faiss

# 파일 로깅용 로거 (상세 정보를 파일에만 기록)
file_logger = logging.getLogger("file_logger")

class RAGRetriever:
    """
    LlamaIndex와 FAISS를 사용하여 RAG 파이프라인을 생성, 로드 및 쿼리합니다.
    지식 베이스 문서를 인덱싱하고 관련 정보를 효율적으로 검색하는 역할을 합니다.
    """

    def __init__(self, knowledge_base_path: str, vector_store_path: str, rebuild: bool = False):
        """
        RAGRetriever를 초기화하고 로컬 임베딩 모델을 설정합니다.

        Args:
            knowledge_base_path (str): 지식 베이스 문서가 포함된 디렉토리 경로.
            vector_store_path (str): FAISS 인덱스를 저장하거나 로드할 디렉토리 경로.
            rebuild (bool): True일 경우 기존 인덱스를 강제로 재생성합니다.
        """
        self.knowledge_base_path = Path(knowledge_base_path)
        self.vector_store_path = Path(vector_store_path)
        self.rebuild = rebuild
        self.index: VectorStoreIndex = None
        
        # 로컬 임베딩 모델 및 청크 분할기 설정 (API 키 필요 없음)
        # LlamaIndex의 전역 설정을 변경할 때 발생하는 불필요한 터미널 출력을 억제
        with suppress_stdout():
            Settings.embed_model = HuggingFaceEmbedding(model_name="jhgan/ko-sroberta-multitask")
            Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
            Settings.llm = None # LLM 호출 비활성화 (순수 RAG 검색만 수행)

    def load(self) -> None:
        """
        기존 FAISS 인덱스를 로드하거나, 존재하지 않을 경우 새로 빌드합니다.
        rebuild 플래그가 True이면 기존 인덱스를 삭제하고 다시 빌드합니다.
        """
        if self.rebuild and self.vector_store_path.exists():
            file_logger.info("Rebuild flag is set. Deleting existing index.")
            shutil.rmtree(self.vector_store_path)

        # docstore.json 존재 여부로 기존 인덱스 확인
        if (self.vector_store_path / "docstore.json").exists():
            file_logger.info("Loading existing index from storage.")
            try:
                with suppress_stdout():  # LlamaIndex 로딩 메시지 숨기기
                    vector_store = FaissVectorStore.from_persist_dir(str(self.vector_store_path))
                    storage_context = StorageContext.from_defaults(
                        vector_store=vector_store, persist_dir=str(self.vector_store_path)
                    )
                    self.index = load_index_from_storage(storage_context=storage_context)
                file_logger.info("Index loaded successfully.")
            except Exception as e:
                file_logger.error(f"Failed to load index, which may be corrupted. Rebuilding... Error: {e}")
                shutil.rmtree(self.vector_store_path)
                self._build_index()
        else:
            file_logger.info("No existing index found. Building a new one...")
            self._build_index()

    def _build_index(self) -> None:
        """
        지식 베이스로부터 FAISS 벡터 인덱스를 빌드하고 디스크에 저장합니다.
        """
        os.makedirs(self.vector_store_path, exist_ok=True)
        documents = SimpleDirectoryReader(str(self.knowledge_base_path)).load_data()
        
        if not documents:
            file_logger.warning(f"No documents found in {self.knowledge_base_path}. Skipping index creation.")
            return

        with suppress_stdout():  # 인덱스 빌드 시 출력 메시지 숨기기
            # FAISS 벡터 스토어 초기화
            d = 768  # 'ko-sroberta-multitask' 모델의 임베딩 차원
            faiss_index = faiss.IndexFlatL2(d)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self.index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
            self.index.storage_context.persist(persist_dir=str(self.vector_store_path))
        
        file_logger.info(f"New index built and saved to {self.vector_store_path}")

    def retrieve_context(self, query_text: str, similarity_top_k: int = 3) -> str:
        """
        로드된 인덱스를 사용하여 쿼리와 관련된 컨텍스트를 검색하여 문자열로 반환합니다.

        Args:
            query_text (str): 사용자 쿼리 문자열.
            similarity_top_k (int): 검색할 상위 결과의 수.

        Returns:
            str: 검색된 컨텍스트 조각들을 합친 단일 문자열.
        
        Raises:
            RuntimeError: 인덱스가 로드되지 않은 경우 발생합니다.
        """
        if self.index is None:
            raise RuntimeError("Index has not been loaded. Please call 'load()' first.")

        with suppress_stdout():  # 검색 시 출력 메시지 숨기기
            retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)
            retrieved_nodes = retriever.retrieve(query_text)
        
        # 검색된 노드의 텍스트를 하나의 문자열로 결합
        return "\n".join([node.get_content() for node in retrieved_nodes])

 