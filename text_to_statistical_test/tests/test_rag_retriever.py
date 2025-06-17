import pytest
import sys
import os

# 테스트 대상 모듈을 import하기 위해 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.components.rag_retriever import RAGRetriever

# NOTE: 이 테스트는 LlamaIndex의 기본 설정을 사용할 경우, 임베딩 및 응답 생성을 위해
# 실제 OpenAI API를 호출할 수 있습니다. 실행 환경에 API 키가 설정되어 있어야 합니다.
def test_retriever_build_and_query(tmp_path):
    """
    RAGRetriever가 임시 파일 시스템의 지식 베이스로부터 인덱스를 빌드하고
    이를 성공적으로 쿼리하는지 테스트합니다.
    """
    # 1. 테스트 설정: 모의 파일 시스템 생성
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    vs_dir = tmp_path / "vs"
    vs_dir.mkdir()

    # 모의 지식 베이스 파일 생성
    (kb_dir / "glossary.md").write_text("고객 만족도는 satisfaction_score를 의미합니다.")

    # 2. 테스트 로직
    # 모의 경로를 사용하여 RAGRetriever 인스턴스화
    retriever = RAGRetriever(knowledge_base_path=str(kb_dir), vector_store_path=str(vs_dir))

    # .load()를 호출하여 인덱스 빌드 트리거
    retriever.load()

    # LlamaIndex의 새로운 저장 방식에 따라 'docstore.json' 파일의 존재를 확인
    assert (vs_dir / "docstore.json").exists()

    # 쿼리를 실행하여 응답 확인
    response = retriever.retrieve_context("고객 만족도란?")

    # 응답에 핵심 키워드가 포함되어 있는지 검증
    assert 'satisfaction_score' in response 