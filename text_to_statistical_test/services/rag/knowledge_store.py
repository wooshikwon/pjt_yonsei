# 파일명: services/rag/knowledge_store.py

import logging
from pathlib import Path
from typing import Dict, Any, List

from .vector_store import VectorStore

logger = logging.getLogger(__name__)

class KnowledgeStore:
    """파일 시스템의 지식 문서를 VectorStore에 로드하는 역할"""

    def __init__(self, vector_store: VectorStore, knowledge_dir: str = "resources/knowledge_base"):
        self.vector_store = vector_store
        self.knowledge_dir = Path(knowledge_dir)

    def ingest_directory(self, sub_dir: str):
        """특정 하위 디렉토리의 모든 문서를 VectorStore에 추가합니다."""
        target_dir = self.knowledge_dir / sub_dir
        if not target_dir.exists():
            logger.warning(f"지식 베이스 디렉토리를 찾을 수 없습니다: {target_dir}")
            return
            
        docs_to_add = []
        for file_path in target_dir.glob("**/*"):
            if file_path.is_file() and file_path.suffix in ['.md', '.txt', '.json']:
                try:
                    text = file_path.read_text(encoding='utf-8')
                    metadata = {'source': str(file_path), 'collection': sub_dir}
                    docs_to_add.append({'text': text, 'metadata': metadata})
                except Exception as e:
                    logger.error(f"파일 읽기 오류: {file_path}, {e}")
        
        if docs_to_add:
            self.vector_store.add_documents(docs_to_add)
            logger.info(f"'{sub_dir}' 컬렉션에서 {len(docs_to_add)}개의 문서를 추가했습니다.")