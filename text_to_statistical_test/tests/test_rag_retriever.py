import pytest
import sys
import os
import shutil
from pathlib import Path

# 테스트 대상 모듈을 import하기 위해 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.components.rag_retriever import RAGRetriever

# NOTE: 이 테스트는 로컬 임베딩 모델(jhgan/ko-sroberta-multitask)을 사용하므로
# OpenAI API 키가 필요하지 않습니다. 모든 처리는 로컬에서 수행됩니다.

@pytest.fixture
def temp_dirs(tmp_path):
    """테스트를 위한 임시 디렉토리 생성 Fixture"""
    kb_dir = tmp_path / "knowledge_base"
    kb_dir.mkdir()
    index_dir = tmp_path / "rag_index"
    index_dir.mkdir()
    
    # 모의 지식 베이스 파일 생성
    (kb_dir / "glossary.md").write_text("첫 번째 문서: 고객 만족도는 satisfaction_score를 의미합니다.", encoding='utf-8')
    
    return str(kb_dir), str(index_dir)

def test_initial_index_build_debug(temp_dirs):
    """
    Case 1: 기존 인덱스가 없을 때 새로 빌드되는지 테스트 (디버깅 정보 포함)
    """
    kb_path, index_path = temp_dirs
    
    print(f"\n=== 테스트 시작: index_path = {index_path} ===")
    
    retriever = RAGRetriever(knowledge_base_path=kb_path, vector_store_path=index_path, rebuild=False)
    retriever.load()
    
    # 디버깅: 생성된 파일 목록 출력
    index_path_obj = Path(index_path)
    if index_path_obj.exists():
        print(f"생성된 파일 목록:")
        for file in index_path_obj.iterdir():
            print(f"  - {file.name} (크기: {file.stat().st_size} bytes)")
    else:
        print("ERROR: index_path 디렉토리가 존재하지 않습니다!")
    
    # 기존 테스트 로직 - 실패 시 더 자세한 정보 출력
    docstore_exists = (index_path_obj / "docstore.json").exists()
    vector_store_exists = (index_path_obj / "default__vector_store.json").exists()
    
    print(f"docstore.json 존재: {docstore_exists}")
    print(f"default__vector_store.json 존재: {vector_store_exists}")
    
    # 적어도 하나는 존재해야 함
    assert docstore_exists or vector_store_exists, f"인덱스 파일이 생성되지 않았습니다. 경로: {index_path}"
    
    # 검색 기능 테스트
    context = retriever.retrieve_context("고객 만족도란?")
    assert "satisfaction_score" in context

def test_simple_build_only(temp_dirs):
    """
    Case 2: 단순히 인덱스 빌드만 테스트 (최소한의 검증)
    """
    kb_path, index_path = temp_dirs
    
    retriever = RAGRetriever(knowledge_base_path=kb_path, vector_store_path=index_path)
    retriever.load()
    
    # 인덱스 객체가 생성되었는지만 확인
    assert retriever.index is not None, "인덱스 객체가 생성되지 않았습니다."
    
    # 최소한 하나의 파일이라도 생성되었는지 확인
    index_files = list(Path(index_path).iterdir())
    assert len(index_files) > 0, f"인덱스 디렉토리에 파일이 생성되지 않았습니다: {index_files}"

def test_loading_existing_index(temp_dirs):
    """
    Case 2: 기존 인덱스를 다시 빌드하지 않고 로드하는지 테스트
    """
    kb_path, index_path = temp_dirs
    
    # 1. 첫 번째 Retriever가 인덱스를 빌드
    retriever1 = RAGRetriever(knowledge_base_path=kb_path, vector_store_path=index_path, rebuild=False)
    retriever1.load()
    
    # 생성 시간 확인을 위해 임의의 파일 수정 시간 기록
    initial_mod_time = (Path(index_path) / "default__vector_store.json").stat().st_mtime

    # 2. 두 번째 Retriever가 기존 인덱스를 로드 (rebuild=False)
    retriever2 = RAGRetriever(knowledge_base_path=kb_path, vector_store_path=index_path, rebuild=False)
    retriever2.load()
    
    loaded_mod_time = (Path(index_path) / "default__vector_store.json").stat().st_mtime
    
    # 파일이 수정되지 않았어야 함 (재빌드 X)
    assert initial_mod_time == loaded_mod_time

def test_forced_rebuild_index(temp_dirs):
    """
    Case 3: rebuild=True일 때 인덱스를 강제로 재생성하는지 테스트
    """
    kb_path, index_path = temp_dirs
    
    # 1. 인덱스 초기 빌드
    retriever1 = RAGRetriever(knowledge_base_path=kb_path, vector_store_path=index_path)
    retriever1.load()
    
    # 2. 지식 베이스 파일 변경
    (Path(kb_path) / "glossary.md").write_text("두 번째 문서: 이제 성과는 performance를 의미합니다.", encoding='utf-8')

    # 3. rebuild=True로 재생성
    retriever2 = RAGRetriever(knowledge_base_path=kb_path, vector_store_path=index_path, rebuild=True)
    retriever2.load()

    # 검색 결과가 새로운 문서 내용을 반영하는지 확인
    context = retriever2.retrieve_context("성과란?")
    assert "performance" in context
    assert "satisfaction_score" not in context 