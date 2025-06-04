"""
LLM Agent 기반 통계 검정 자동화 시스템 메인 실행 스크립트

이 스크립트는 전체 시스템의 진입점으로, 의존성 주입 및 워크플로우 실행을 담당합니다.
"""

import argparse
import logging
import sys
from pathlib import Path

from config.settings import (
    LLM_PROVIDER, LLM_MODEL_NAME, WORKFLOW_FILE_PATH,
    CODE_SNIPPETS_DIR, RAG_INDEX_PATH, INPUT_DATA_DEFAULT_DIR,
    OUTPUT_RESULTS_DIR, LOG_LEVEL
)
from core.agent import LLMAgent
from core.workflow_manager import WorkflowManager
from core.decision_engine import DecisionEngine
from core.context_manager import ContextManager
from llm_services.llm_client import LLMClient
from llm_services.prompt_crafter import PromptCrafter
from data_processing.data_loader import DataLoader
from rag_system.code_retriever import CodeRetriever
from code_execution.safe_code_executor import SafeCodeExecutor
from reporting.report_generator import ReportGenerator


def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    

def setup_dependencies() -> dict:
    """
    각 서비스의 인스턴스를 생성하고 의존성을 주입합니다.
    
    Returns:
        dict: 초기화된 서비스 인스턴스들의 딕셔너리
    """
    logging.info("의존성 초기화 중...")
    
    # 1. LLM Client 초기화
    llm_client = LLMClient(
        provider_name=LLM_PROVIDER,
        model_name=LLM_MODEL_NAME
    )
    
    # 2. Workflow Manager 초기화
    workflow_manager = WorkflowManager(WORKFLOW_FILE_PATH)
    
    # 3. Decision Engine 초기화
    decision_engine = DecisionEngine()
    
    # 4. Context Manager 초기화
    context_manager = ContextManager(llm_client)
    
    # 5. Prompt Crafter 초기화
    prompt_crafter = PromptCrafter("llm_services/prompts")
    
    # 6. Data Loader 초기화
    data_loader = DataLoader()
    
    # 7. Code Retriever 초기화
    code_retriever = CodeRetriever(CODE_SNIPPETS_DIR)
    
    # 8. Safe Code Executor 초기화
    safe_code_executor = SafeCodeExecutor()
    
    # 9. Report Generator 초기화
    report_generator = ReportGenerator(OUTPUT_RESULTS_DIR)
    
    dependencies = {
        'llm_client': llm_client,
        'workflow_manager': workflow_manager,
        'decision_engine': decision_engine,
        'context_manager': context_manager,
        'prompt_crafter': prompt_crafter,
        'data_loader': data_loader,
        'code_retriever': code_retriever,
        'safe_code_executor': safe_code_executor,
        'report_generator': report_generator
    }
    
    logging.info("의존성 초기화 완료")
    return dependencies


def run_agent_workflow(dependencies: dict, input_data_path: str) -> str:
    """
    LLMAgent 인스턴스를 생성하고 분석 워크플로우를 시작합니다.
    
    Args:
        dependencies: 초기화된 서비스 인스턴스들
        input_data_path: 입력 데이터 파일 경로
        
    Returns:
        str: 생성된 보고서 파일 경로
    """
    logging.info("Agent 워크플로우 시작")
    
    # LLMAgent 인스턴스 생성
    agent = LLMAgent(
        workflow_manager=dependencies['workflow_manager'],
        decision_engine=dependencies['decision_engine'],
        context_manager=dependencies['context_manager'],
        llm_client=dependencies['llm_client'],
        prompt_crafter=dependencies['prompt_crafter'],
        data_loader=dependencies['data_loader'],
        code_retriever=dependencies['code_retriever'],
        safe_code_executor=dependencies['safe_code_executor'],
        report_generator=dependencies['report_generator']
    )
    
    # 워크플로우 실행
    report_path = agent.run(input_data_path)
    
    logging.info(f"워크플로우 완료. 보고서: {report_path}")
    return report_path


def main():
    """메인 함수: CLI 인자 파싱 및 전체 실행 흐름 제어"""
    parser = argparse.ArgumentParser(
        description="LLM Agent 기반 통계 검정 자동화 시스템"
    )
    parser.add_argument(
        '--input-data', 
        type=str, 
        help='입력 데이터 파일 경로 (Tableau .hyper, CSV 등)',
        default=None
    )
    parser.add_argument(
        '--query',
        type=str,
        help='분석 요청 (자연어)',
        default=None
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='대화형 모드로 실행'
    )
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging()
    
    try:
        # 의존성 초기화
        dependencies = setup_dependencies()
        
        # 입력 데이터 경로 처리
        if args.input_data:
            input_data_path = args.input_data
        else:
            # 기본 경로에서 데이터 파일 검색 또는 대화형 입력
            input_data_path = None
            
        # 분석 요청 처리
        if args.query:
            # CLI에서 직접 분석 요청이 주어진 경우
            print(f"분석 요청: {args.query}")
            
        # 워크플로우 실행
        report_path = run_agent_workflow(dependencies, input_data_path)
        
        print(f"\n✅ 분석 완료!")
        print(f"📊 결과 보고서: {report_path}")
        
    except Exception as e:
        logging.error(f"실행 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 