# 파일명: services/rag/rag_service.py
# 이 파일은 LanceDB와 LangChain을 사용하여 RAG(검색 증강 생성) 기능을 제공하는
# 현대적이고 효율적인 서비스입니다.

import lancedb
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from langchain_community.vectorstores import LanceDB
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from lancedb.pydantic import LanceModel, Vector

from config.settings import get_settings
from utils import RAGException, Singleton

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class RAGService(metaclass=Singleton):
    """
    지식 베이스 문서의 수집, 저장, 검색을 담당하는 RAG 서비스입니다.
    LanceDB를 벡터 저장소로, SentenceTransformer를 임베딩 모델로 사용합니다.
    """
    
    TABLE_NAME = "knowledge_base"
    _schema = None

    def __init__(self, 
                 knowledge_base_dir: str = "resources/knowledge_base",
                 rag_storage_path: str = "output_data/rag_storage"):
        
        logger.info("RAG 서비스 초기화를 시작합니다...")
        self.knowledge_dir = Path(knowledge_base_dir)
        self.db_path = Path(rag_storage_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        try:
            self.db = lancedb.connect(self.db_path)
            self._initialize_embedding_model()
            self._initialize_text_splitter()

            # 스키마 클래스를 동적으로 생성
            class KnowledgeSchema(LanceModel):
                vector: Vector(self.vector_dim)
                text: str
                collection: str
                source: str
            
            self._schema = KnowledgeSchema

            self.table = self._get_or_create_table()
            self.is_initialized = True
            logger.info("RAG 서비스가 성공적으로 초기화되었습니다.")
        except Exception as e:
            self.is_initialized = False
            logger.error(f"RAG 서비스 초기화 중 심각한 오류 발생: {e}", exc_info=True)
            raise RAGException("RAGService 초기화 실패") from e

    def _initialize_embedding_model(self, model_name: str = "all-MiniLM-L6-v2"):
        """임베딩 모델을 로드합니다."""
        logger.info(f"임베딩 모델 '{model_name}' 로드 중...")
        self.embedding_model = SentenceTransformer(model_name)
        self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info("임베딩 모델 로드 완료.")

    def _initialize_text_splitter(self):
        """텍스트 분할기를 초기화합니다."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "##", "#", " ", ""],
            length_function=len,
        )

    def _get_or_create_table(self):
        """LanceDB에서 테이블을 가져오거나 새로 생성합니다."""
        if self.TABLE_NAME in self.db.table_names():
            logger.info(f"기존 LanceDB 테이블 '{self.TABLE_NAME}'을 로드합니다.")
            return self.db.open_table(self.TABLE_NAME)
        
        logger.info(f"새로운 LanceDB 테이블 '{self.TABLE_NAME}'을 생성합니다.")
        return self.db.create_table(self.TABLE_NAME, schema=self._schema, mode="overwrite")

    def ingest_knowledge_base(self, force_reingest: bool = False):
        """
        knowledge_base 디렉토리의 모든 문서를 처리하여 벡터 저장소에 추가합니다.
        이미 데이터가 있는 경우, force_reingest가 True가 아니면 실행하지 않습니다.
        """
        if self.table.to_pandas().shape[0] > 0 and not force_reingest:
            logger.info("지식 베이스가 이미 수집되어 있습니다. 재수집을 원하시면 force_reingest=True로 설정하세요.")
            return

        logger.info(f"'{self.knowledge_dir}'에서 지식 베이스 수집을 시작합니다...")
        
        all_chunks = []
        for file_path in self.knowledge_dir.glob("**/*.md"):
            collection_name = file_path.parent.name
            logger.debug(f"파일 처리 중: {file_path} (컬렉션: {collection_name})")
            
            try:
                content = file_path.read_text(encoding='utf-8')
                chunks = self.text_splitter.split_text(content)
                
                for chunk in chunks:
                    all_chunks.append({
                        "text": chunk,
                        "collection": collection_name,
                        "source": str(file_path.relative_to(self.knowledge_dir))
                    })
            except Exception as e:
                logger.error(f"파일 처리 실패: {file_path}, 오류: {e}")

        if not all_chunks:
            logger.warning("수집할 문서를 찾지 못했습니다.")
            return
            
        logger.info(f"총 {len(all_chunks)}개의 청크를 임베딩 및 추가합니다...")
        
        # 텍스트 목록을 추출하여 일괄적으로 임베딩
        texts_to_embed = [chunk['text'] for chunk in all_chunks]
        vectors = self.embedding_model.encode(texts_to_embed, show_progress_bar=True)
        
        # 각 청크에 벡터 추가
        for i, chunk in enumerate(all_chunks):
            chunk['vector'] = vectors[i]
        
        # 데이터 일괄 추가. lancedb가 pydantic 모델에 맞춰 자동으로 변환합니다.
        self.table.add(all_chunks)
        
        logger.info("지식 베이스 수집 및 벡터 저장소 추가 완료.")
        
        # FTS 인덱스 생성
        try:
            logger.info("Full-text search (FTS) 인덱스를 생성합니다...")
            self.table.create_fts_index("text")
            logger.info("FTS 인덱스 생성 완료.")
        except Exception as e:
            logger.warning(f"FTS 인덱스 생성 실패: {e}. FTS 검색을 사용할 수 없습니다.")

    def search(self, query: str, top_k: int = 5, collection: Optional[Union[str, List[str]]] = None) -> List[Dict[str, Any]]:
        """
        벡터 검색을 수행합니다. 특정 컬렉션으로 범위를 제한할 수 있습니다.
        """
        if not self.is_initialized:
            raise RAGException("RAGService가 초기화되지 않았습니다.")
            
        query_vector = self.embedding_model.encode([query])[0]
        
        search_request = self.table.search(query_vector).limit(top_k)
        
        # 컬렉션 필터링
        if collection:
            if isinstance(collection, str):
                filter_query = f"collection = '{collection}'"
            else: # list of strings
                filter_query = " OR ".join([f"collection = '{c}'" for c in collection])
            search_request = search_request.where(filter_query)
        
        results = search_request.to_df().to_dict('records')
        logger.info(f"'{query}'에 대한 검색 완료. {len(results)}개 결과 반환.")
        return results

    @staticmethod
    def build_context_from_results(results: List[Dict[str, Any]], max_length: int = 4000) -> str:
        """검색 결과를 LLM 프롬프트에 주입할 컨텍스트 문자열로 변환합니다."""
        context = ""
        for result in results:
            source_info = f"[출처: {result['source']}, 컬렉션: {result['collection']}]"
            chunk_content = result['text']
            
            # 길이 제한 확인
            if len(context) + len(source_info) + len(chunk_content) + 5 > max_length:
                break
            
            context += f"--- 문서 조각 ---\n"
            context += f"{source_info}\n"
            context += f"{chunk_content}\n\n"
            
        logger.info(f"컨텍스트 구성 완료. 최종 길이: {len(context)}")
        return context.strip()

# 전역 RAG 서비스 인스턴스 (옵션)
# 애플리케이션의 다른 부분에서 쉽게 접근할 수 있도록 함
try:
    rag_service_instance = RAGService()
except RAGException as e:
    # 애플리케이션 시작 시 로거가 이미 설정되어 있으므로 print 대신 logger 사용
    logging.getLogger(__name__).critical(f"전역 RAG 서비스 인스턴스 생성 실패: {e}")
    rag_service_instance = None 