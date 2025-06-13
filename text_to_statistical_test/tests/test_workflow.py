import pytest
import asyncio
from pathlib import Path
import os
import pandas as pd

# 테스트 대상 모듈 임포트 전에 경로 설정
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pipeline.orchestrator import Orchestrator
from config.settings import get_settings
from services.rag.rag_service import RAGService

# 테스트를 위한 전역 설정
SAMPLE_CSV_PATH = "input_data/data_files/sample_customers.csv"
TEST_OUTPUT_DIR = "output_data/test_run"

@pytest.fixture(scope="session")
def event_loop():
    """pytest-asyncio를 위한 이벤트 루프 제공"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """테스트 실행 전반에 걸쳐 필요한 환경을 설정합니다."""
    # .env 파일 로드
    from dotenv import load_dotenv
    load_dotenv()

    # 필수 디렉토리 생성
    Path(TEST_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # 샘플 데이터 파일 생성
    if not Path(SAMPLE_CSV_PATH).exists():
        Path(SAMPLE_CSV_PATH).parent.mkdir(parents=True, exist_ok=True)
        sample_data = {
            'CustomerID': range(1, 101),
            'Age': [25, 34, 45, 23, 54] * 20,
            'Gender': ['Male', 'Female'] * 50,
            'SubscriptionTier': ['Basic', 'Premium', 'Standard', 'Basic', 'Premium'] * 20,
            'MonthlySpend': [50, 120, 80, 45, 130] * 20,
            'UsedSupport': [0, 1, 1, 0, 1] * 20
        }
        pd.DataFrame(sample_data).to_csv(SAMPLE_CSV_PATH, index=False)

    # RAG 서비스 초기화 및 지식 베이스 수집
    print("Setting up RAG service for tests...")
    try:
        rag_service = RAGService(rag_storage_path=f"{TEST_OUTPUT_DIR}/rag_storage")
        rag_service.ingest_knowledge_base(force_reingest=True)
        print("RAG service setup complete.")
    except Exception as e:
        pytest.fail(f"RAG service initialization failed: {e}")

@pytest.mark.asyncio
async def test_full_workflow_run():
    """
    Orchestrator의 전체 워크플로우를 실행하는 통합 테스트.
    간단한 T-검정 요청으로 파이프라인이 끝까지 실행되고 보고서를 생성하는지 확인합니다.
    """
    # GIVEN: 유효한 데이터 파일 경로와 분석 요청
    file_path = SAMPLE_CSV_PATH
    user_request = "고객 지원(UsedSupport) 사용 여부에 따라 월 지출액(MonthlySpend)에 차이가 있는지 t-검정으로 분석해줘."
    
    assert Path(file_path).exists(), "테스트를 위한 샘플 CSV 파일이 없습니다."
    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY가 설정되지 않았습니다."

    # WHEN: Orchestrator의 run 메소드를 실행
    orchestrator = Orchestrator()
    final_context = await orchestrator.run(
        file_path=file_path,
        user_request=user_request
    )

    # THEN: 최종 컨텍스트가 유효하고, 결과물이 정상적으로 생성되었는지 확인
    assert isinstance(final_context, dict), "Orchestrator가 유효한 컨텍스트(dict)를 반환하지 않았습니다."
    
    final_report_path = final_context.get("final_report_path")
    final_report_content = final_context.get("final_report_content")

    # 1. 경로와 콘텐츠가 존재하는지 확인
    assert isinstance(final_report_path, str), "최종 보고서 경로가 문자열이 아닙니다."
    assert isinstance(final_report_content, str), "최종 보고서 내용이 문자열이 아닙니다."
    
    # 2. 파일 시스템에서 보고서 파일이 실제로 생성되었는지 확인
    assert Path(final_report_path).exists(), f"생성된 보고서 파일을 찾을 수 없습니다: {final_report_path}"
    assert Path(final_report_path).name.startswith("report_"), "보고서 파일명이 'report_'로 시작하지 않습니다."
    assert Path(final_report_path).suffix == ".md", "보고서 파일이 .md 형식이 아닙니다."

    # 3. 생성된 Markdown 보고서의 내용을 검증
    assert final_report_content.startswith("#"), "보고서가 Markdown 제목('#')으로 시작하지 않습니다."
    assert "## Key Findings" in final_report_content, "보고서에 'Key Findings' 섹션이 포함되지 않았습니다."
    assert "## Conclusion & Recommendations" in final_report_content, "보고서에 'Conclusion & Recommendations' 섹션이 포함되지 않았습니다."
    assert "## Detailed Steps" in final_report_content, "보고서에 'Detailed Steps' 섹션이 포함되지 않았습니다."
    assert "t-test" in final_report_content.lower(), "보고서 내용에 't-test' 분석에 대한 언급이 없습니다." 