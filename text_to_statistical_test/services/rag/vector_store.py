# 파일명: services/rag/vector_store.py

import logging
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any, Optional

# [UTIL-REQ] error_handler.py 및 helpers.py의 함수/클래스가 필요합니다.
from utils.error_handler import RAGException
from utils.helpers import generate_unique_id, safe_json_dumps, safe_json_loads

logger = logging.getLogger(__name__)

class VectorStore:
    """문서 벡터화 및 유사도 검색을 위한 벡터 저장소 (FAISS 및 SentenceTransformer 기반)"""
    
    def __init__(self, storage_path: str = "output_data/vector_store", model_name: str = "all-MiniLM-L6-v2"):
        self.storage_path = Path(storage_path)
        self.model_name = model_name
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        try:
            self.embedding_model = SentenceTransformer(model_name)
            self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()
        except Exception as e:
            raise RAGException(f"임베딩 모델 로드 실패: {e}")

        self.index_file = self.storage_path / "faiss_index.bin"
        self.metadata_file = self.storage_path / "metadata.json"
        
        self.doc_metadata: Dict[str, Dict] = {} # doc_id -> {text, metadata, faiss_id}
        self.faiss_id_to_doc_id: Dict[int, str] = {}
        
        self._load()

    def add_documents(self, docs: List[Dict[str, Any]]) -> List[str]:
        """문서들을 벡터 저장소에 추가합니다."""
        if not docs:
            return []
        
        texts = [doc['text'] for doc in docs]
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        embeddings_normalized = self._normalize_vectors(embeddings)
        
        doc_ids = []
        new_faiss_ids = []
        
        for i, doc in enumerate(docs):
            doc_id = generate_unique_id("doc")
            faiss_id = self.index.ntotal + i
            
            doc_ids.append(doc_id)
            new_faiss_ids.append(faiss_id)
            
            self.doc_metadata[doc_id] = {
                'text': doc['text'],
                'metadata': doc.get('metadata', {}),
                'faiss_id': faiss_id
            }
            self.faiss_id_to_doc_id[faiss_id] = doc_id
            
        self.index.add(embeddings_normalized)
        self._save()
        logger.info(f"{len(docs)}개의 문서가 벡터 저장소에 추가되었습니다.")
        return doc_ids

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """쿼리와 유사한 문서를 검색합니다."""
        if self.index.ntotal == 0:
            return []
            
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        query_normalized = self._normalize_vectors(query_embedding)
        
        scores, faiss_ids = self.index.search(query_normalized, top_k)
        
        results = []
        for i, faiss_id in enumerate(faiss_ids[0]):
            if faiss_id == -1: continue
            
            doc_id = self.faiss_id_to_doc_id.get(faiss_id)
            if doc_id and doc_id in self.doc_metadata:
                doc = self.doc_metadata[doc_id]
                results.append({
                    'doc_id': doc_id,
                    'text': doc['text'],
                    'metadata': doc['metadata'],
                    'score': float(scores[0][i])
                })
        return results

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """코사인 유사도 검색을 위해 벡터를 정규화합니다."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms!=0)

    def _save(self):
        """인덱스와 메타데이터를 파일에 저장합니다."""
        faiss.write_index(self.index, str(self.index_file))
        metadata_to_save = {
            'doc_metadata': self.doc_metadata,
            'faiss_id_to_doc_id': self.faiss_id_to_doc_id
        }
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            f.write(safe_json_dumps(metadata_to_save))
    
    def _load(self):
        """파일로부터 인덱스와 메타데이터를 로드합니다."""
        if self.index_file.exists() and self.metadata_file.exists():
            self.index = faiss.read_index(str(self.index_file))
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadata = safe_json_loads(f.read())
                self.doc_metadata = metadata.get('doc_metadata', {})
                # JSON 키는 문자열이므로 faiss_id(정수)로 변환
                self.faiss_id_to_doc_id = {int(k): v for k, v in metadata.get('faiss_id_to_doc_id', {}).items()}
            logger.info(f"기존 벡터 저장소 로드 완료: {self.index.ntotal}개 문서")
        else:
            self.index = faiss.IndexFlatIP(self.vector_dim)
            logger.info("새로운 벡터 저장소를 생성합니다.")