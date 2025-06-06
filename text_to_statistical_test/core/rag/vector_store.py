"""
Vector Store

RAG 시스템을 위한 벡터 저장소
문서 임베딩 생성, 저장, 유사도 검색 기능 제공
"""

import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
import pickle

from utils.global_cache import get_global_cache
from utils.helpers import generate_unique_id, safe_json_dumps, safe_json_loads
from utils.error_handler import ErrorHandler, RAGError

logger = logging.getLogger(__name__)

class VectorStore:
    """
    문서 벡터화 및 유사도 검색을 위한 벡터 저장소
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 vector_dim: int = 384,
                 storage_path: Optional[str] = None):
        """
        벡터 저장소 초기화
        
        Args:
            model_name: 임베딩 모델명
            vector_dim: 벡터 차원
            storage_path: 저장소 경로 (None이면 메모리만 사용)
        """
        self.model_name = model_name
        self.vector_dim = vector_dim
        self.storage_path = Path(storage_path) if storage_path else None
        
        # 임베딩 모델 로드
        try:
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"임베딩 모델 로드 완료: {model_name}")
        except Exception as e:
            logger.error(f"임베딩 모델 로드 실패: {str(e)}")
            raise RAGError(f"임베딩 모델 로드 실패: {str(e)}")
        
        # FAISS 인덱스 초기화
        self.index = faiss.IndexFlatIP(vector_dim)  # Inner Product (코사인 유사도)
        self.document_metadata = {}  # 문서 메타데이터
        self.id_to_doc = {}  # ID -> 문서 매핑
        self.next_id = 0
        
        # 캐시 및 오류 처리
        self.cache = get_global_cache()
        self.error_handler = ErrorHandler()
        
        # 저장소 경로 설정
        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._load_index()
        
        logger.info("벡터 저장소 초기화 완료")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        문서들을 벡터 저장소에 추가
        
        Args:
            documents: 문서 리스트. 각 문서는 {'text': str, 'metadata': dict} 형태
            
        Returns:
            List[str]: 추가된 문서들의 ID 리스트
        """
        try:
            doc_ids = []
            texts = []
            metadatas = []
            
            # 문서 전처리
            for doc in documents:
                doc_id = generate_unique_id("doc")
                doc_ids.append(doc_id)
                
                text = doc.get('text', '')
                metadata = doc.get('metadata', {})
                
                texts.append(text)
                metadatas.append(metadata)
                
                # 문서 메타데이터 저장
                self.document_metadata[doc_id] = {
                    'text': text,
                    'metadata': metadata,
                    'id': self.next_id,
                    'added_at': np.datetime64('now')
                }
                
                self.id_to_doc[self.next_id] = doc_id
                self.next_id += 1
            
            # 텍스트 임베딩 생성
            logger.info(f"{len(texts)}개 문서의 임베딩 생성 중...")
            embeddings = self._generate_embeddings(texts)
            
            # FAISS 인덱스에 추가
            embeddings_normalized = self._normalize_vectors(embeddings)
            self.index.add(embeddings_normalized)
            
            # 저장소에 저장
            if self.storage_path:
                self._save_index()
            
            logger.info(f"{len(doc_ids)}개 문서가 벡터 저장소에 추가되었습니다.")
            return doc_ids
            
        except Exception as e:
            logger.error(f"문서 추가 실패: {str(e)}")
            raise RAGError(f"문서 추가 실패: {str(e)}")
    
    def search(self, 
               query: str, 
               k: int = 5,
               filter_metadata: Optional[Dict[str, Any]] = None,
               min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        유사도 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            filter_metadata: 메타데이터 필터링 조건
            min_score: 최소 유사도 점수
            
        Returns:
            List[Dict[str, Any]]: 검색 결과 리스트
        """
        try:
            if self.index.ntotal == 0:
                logger.warning("벡터 저장소가 비어있습니다.")
                return []
            
            # 쿼리 임베딩 생성
            query_embedding = self._generate_embeddings([query])
            query_normalized = self._normalize_vectors(query_embedding)
            
            # 유사도 검색
            scores, indices = self.index.search(query_normalized, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # 유효하지 않은 인덱스
                    continue
                
                if score < min_score:
                    continue
                
                doc_id = self.id_to_doc.get(idx)
                if not doc_id:
                    continue
                
                doc_metadata = self.document_metadata.get(doc_id)
                if not doc_metadata:
                    continue
                
                # 메타데이터 필터링
                if filter_metadata and not self._match_metadata_filter(
                    doc_metadata['metadata'], filter_metadata):
                    continue
                
                result = {
                    'doc_id': doc_id,
                    'text': doc_metadata['text'],
                    'metadata': doc_metadata['metadata'],
                    'score': float(score),
                    'index': int(idx)
                }
                results.append(result)
            
            logger.debug(f"검색 완료: {len(results)}개 결과 반환")
            return results
            
        except Exception as e:
            logger.error(f"검색 실패: {str(e)}")
            raise RAGError(f"검색 실패: {str(e)}")
    
    def delete_document(self, doc_id: str) -> bool:
        """
        문서 삭제
        
        Args:
            doc_id: 문서 ID
            
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            if doc_id not in self.document_metadata:
                logger.warning(f"문서를 찾을 수 없습니다: {doc_id}")
                return False
            
            # 메타데이터에서 제거
            doc_meta = self.document_metadata.pop(doc_id)
            idx = doc_meta['id']
            self.id_to_doc.pop(idx, None)
            
            # FAISS 인덱스에서는 직접 삭제가 어려우므로
            # 메타데이터만 제거하고 실제 재구축은 별도 메서드로 제공
            
            logger.info(f"문서 삭제됨: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"문서 삭제 실패: {str(e)}")
            return False
    
    def update_document(self, doc_id: str, new_text: str, 
                       new_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        문서 업데이트
        
        Args:
            doc_id: 문서 ID
            new_text: 새로운 텍스트
            new_metadata: 새로운 메타데이터
            
        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            if doc_id not in self.document_metadata:
                logger.warning(f"문서를 찾을 수 없습니다: {doc_id}")
                return False
            
            # 기존 문서 정보
            old_meta = self.document_metadata[doc_id]
            idx = old_meta['id']
            
            # 새로운 임베딩 생성
            new_embedding = self._generate_embeddings([new_text])
            new_embedding_normalized = self._normalize_vectors(new_embedding)
            
            # FAISS 인덱스 업데이트 (재구축 필요)
            # 현재는 메타데이터만 업데이트
            self.document_metadata[doc_id].update({
                'text': new_text,
                'metadata': new_metadata or old_meta['metadata'],
                'updated_at': np.datetime64('now')
            })
            
            logger.info(f"문서 업데이트됨: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"문서 업데이트 실패: {str(e)}")
            return False
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        문서 조회
        
        Args:
            doc_id: 문서 ID
            
        Returns:
            Optional[Dict[str, Any]]: 문서 정보 (없으면 None)
        """
        return self.document_metadata.get(doc_id)
    
    def list_documents(self, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        문서 목록 조회
        
        Args:
            filter_metadata: 메타데이터 필터링 조건
            
        Returns:
            List[Dict[str, Any]]: 문서 목록
        """
        results = []
        for doc_id, doc_meta in self.document_metadata.items():
            if filter_metadata and not self._match_metadata_filter(
                doc_meta['metadata'], filter_metadata):
                continue
            
            result = {
                'doc_id': doc_id,
                'text_preview': doc_meta['text'][:200] + '...' if len(doc_meta['text']) > 200 else doc_meta['text'],
                'metadata': doc_meta['metadata'],
                'added_at': str(doc_meta.get('added_at', '')),
                'text_length': len(doc_meta['text'])
            }
            results.append(result)
        
        return results
    
    def rebuild_index(self) -> bool:
        """
        인덱스 재구축 (삭제된 문서 반영)
        
        Returns:
            bool: 재구축 성공 여부
        """
        try:
            logger.info("인덱스 재구축 시작...")
            
            # 새로운 인덱스 생성
            new_index = faiss.IndexFlatIP(self.vector_dim)
            new_id_to_doc = {}
            
            # 유효한 문서들만 다시 추가
            texts = []
            doc_ids = []
            
            for doc_id, doc_meta in self.document_metadata.items():
                texts.append(doc_meta['text'])
                doc_ids.append(doc_id)
            
            if texts:
                # 임베딩 재생성
                embeddings = self._generate_embeddings(texts)
                embeddings_normalized = self._normalize_vectors(embeddings)
                
                # 새 인덱스에 추가
                new_index.add(embeddings_normalized)
                
                # ID 매핑 재구축
                for i, doc_id in enumerate(doc_ids):
                    new_id_to_doc[i] = doc_id
                    self.document_metadata[doc_id]['id'] = i
            
            # 기존 인덱스 교체
            self.index = new_index
            self.id_to_doc = new_id_to_doc
            self.next_id = len(doc_ids)
            
            # 저장
            if self.storage_path:
                self._save_index()
            
            logger.info("인덱스 재구축 완료")
            return True
            
        except Exception as e:
            logger.error(f"인덱스 재구축 실패: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        벡터 저장소 통계 정보
        
        Returns:
            Dict[str, Any]: 통계 정보
        """
        return {
            'total_documents': len(self.document_metadata),
            'vector_dimension': self.vector_dim,
            'model_name': self.model_name,
            'index_size': self.index.ntotal,
            'storage_path': str(self.storage_path) if self.storage_path else None,
            'next_id': self.next_id
        }
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """텍스트 임베딩 생성"""
        try:
            # 캐시 확인
            cache_key = f"embeddings_{hash(str(texts))}"
            cached_embeddings = self.cache.get(cache_key)
            if cached_embeddings is not None:
                return cached_embeddings
            
            # 임베딩 생성
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            
            # 캐시 저장
            self.cache.set(cache_key, embeddings, ttl=3600)  # 1시간 캐시
            
            return embeddings
            
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {str(e)}")
            raise RAGError(f"임베딩 생성 실패: {str(e)}")
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """벡터 정규화 (코사인 유사도를 위해)"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 0으로 나누기 방지
        return vectors / norms
    
    def _match_metadata_filter(self, doc_metadata: Dict[str, Any], 
                             filter_metadata: Dict[str, Any]) -> bool:
        """메타데이터 필터링 매칭"""
        for key, value in filter_metadata.items():
            if key not in doc_metadata:
                return False
            if doc_metadata[key] != value:
                return False
        return True
    
    def _save_index(self):
        """인덱스를 파일에 저장"""
        if not self.storage_path:
            return
        
        try:
            # FAISS 인덱스 저장
            index_path = self.storage_path / "faiss_index.bin"
            faiss.write_index(self.index, str(index_path))
            
            # 메타데이터 저장
            metadata_path = self.storage_path / "metadata.json"
            metadata = {
                'document_metadata': self.document_metadata,
                'id_to_doc': self.id_to_doc,
                'next_id': self.next_id,
                'model_name': self.model_name,
                'vector_dim': self.vector_dim
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write(safe_json_dumps(metadata))
            
            logger.debug("인덱스 저장 완료")
            
        except Exception as e:
            logger.error(f"인덱스 저장 실패: {str(e)}")
    
    def _load_index(self):
        """파일에서 인덱스 로드"""
        if not self.storage_path:
            return
        
        try:
            index_path = self.storage_path / "faiss_index.bin"
            metadata_path = self.storage_path / "metadata.json"
            
            # 인덱스 파일 존재 확인
            if not index_path.exists() or not metadata_path.exists():
                logger.info("기존 인덱스가 없습니다. 새로 생성합니다.")
                return
            
            # FAISS 인덱스 로드
            self.index = faiss.read_index(str(index_path))
            
            # 메타데이터 로드
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = safe_json_loads(f.read())
            
            if metadata:
                self.document_metadata = metadata.get('document_metadata', {})
                self.id_to_doc = {int(k): v for k, v in metadata.get('id_to_doc', {}).items()}
                self.next_id = metadata.get('next_id', 0)
                
                # 모델 설정 검증
                saved_model = metadata.get('model_name')
                saved_dim = metadata.get('vector_dim')
                
                if saved_model != self.model_name:
                    logger.warning(f"저장된 모델({saved_model})과 현재 모델({self.model_name})이 다릅니다.")
                
                if saved_dim != self.vector_dim:
                    logger.warning(f"저장된 차원({saved_dim})과 현재 차원({self.vector_dim})이 다릅니다.")
            
            logger.info(f"인덱스 로드 완료: {len(self.document_metadata)}개 문서")
            
        except Exception as e:
            logger.error(f"인덱스 로드 실패: {str(e)}")
            # 오류가 발생하면 새로운 인덱스로 시작
            self.index = faiss.IndexFlatIP(self.vector_dim)
            self.document_metadata = {}
            self.id_to_doc = {}
            self.next_id = 0