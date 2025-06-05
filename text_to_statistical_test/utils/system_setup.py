"""
System Setup Utilities

Enhanced RAG 시스템 기반 시스템 초기화, 의존성 설정, 로깅 설정 등을 담당하는 유틸리티 함수들
"""

import logging
import sys
from pathlib import Path

from config.settings import (
    LLM_PROVIDER, LLM_MODEL_NAME, WORKFLOW_FILE_PATH,
    OUTPUT_RESULTS_DIR, LOG_LEVEL, 
    validate_settings, ensure_directories
)
from core.agent import LLMAgent
from core.workflow_manager import WorkflowManager
from core.decision_engine import DecisionEngine
from core.context_manager import ContextManager
from llm_services.llm_client import LLMClient
from llm_services.prompt_crafter import PromptCrafter
from data_processing.data_loader import DataLoader
from code_execution.safe_code_executor import SafeCodeExecutor
from reporting.report_generator import ReportGenerator

# Enhanced RAG 시스템 컴포넌트 import
from rag_system.business_retriever import BusinessRetriever
from rag_system.schema_retriever import SchemaRetriever
from rag_system.rag_manager import RAGManager

# AI 추천 엔진 import
from utils.analysis_recommender import AnalysisRecommender


def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/app.log', encoding='utf-8')
        ]
    )


def setup_dependencies() -> dict:
    """
    Enhanced RAG 시스템 기반으로 각 서비스의 인스턴스를 생성하고 의존성을 주입합니다.
    
    Returns:
        dict: 초기화된 서비스 인스턴스들의 딕셔너리
    """
    logging.info("Enhanced RAG 기반 의존성 초기화 중...")
    
    # 환경 설정 검증
    try:
        validate_settings()
        ensure_directories()
    except Exception as e:
        logging.error(f"환경 설정 검증 실패: {e}")
        raise
    
    # 1. LLM Client 초기화
    try:
        llm_client = LLMClient(
            provider_name=LLM_PROVIDER,
            model_name=LLM_MODEL_NAME
        )
        logging.info(f"LLM 클라이언트 초기화 완료: {LLM_PROVIDER} - {LLM_MODEL_NAME}")
    except Exception as e:
        logging.error(f"LLM 클라이언트 초기화 실패: {e}")
        raise
    
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
    
    # 7. Enhanced RAG 시스템 초기화
    try:
        # RAGManager가 metadata_path와 schema_path를 받도록 수정
        rag_manager = RAGManager(
            metadata_path="input_data/metadata",
            schema_path="input_data/metadata/database_schemas"
        )
        # RAGManager 내부에서 생성된 retriever들을 가져오기
        business_retriever = rag_manager.business_retriever
        schema_retriever = rag_manager.schema_retriever
        logging.info("Enhanced RAG 시스템 초기화 완료")
    except Exception as e:
        logging.error(f"Enhanced RAG 시스템 초기화 실패: {e}")
        raise
    
    # 8. AI 추천 엔진 초기화
    analysis_recommender = AnalysisRecommender(llm_client, prompt_crafter)
    
    # 9. Safe Code Executor 초기화
    safe_code_executor = SafeCodeExecutor()
    
    # 10. Report Generator 초기화
    report_generator = ReportGenerator(OUTPUT_RESULTS_DIR)
    
    dependencies = {
        'llm_client': llm_client,
        'workflow_manager': workflow_manager,
        'decision_engine': decision_engine,
        'context_manager': context_manager,
        'prompt_crafter': prompt_crafter,
        'data_loader': data_loader,
        # Enhanced RAG 시스템 컴포넌트들
        'business_retriever': business_retriever,
        'schema_retriever': schema_retriever,
        'rag_manager': rag_manager,
        'analysis_recommender': analysis_recommender,
        'safe_code_executor': safe_code_executor,
        'report_generator': report_generator
    }
    
    logging.info("모든 Enhanced RAG 기반 의존성 초기화 완료")
    return dependencies


def create_agent_instance(dependencies: dict) -> LLMAgent:
    """
    Enhanced RAG 기반 Agent 인스턴스 생성
    
    Args:
        dependencies: 초기화된 서비스 인스턴스들
        
    Returns:
        LLMAgent: 생성된 Enhanced RAG Agent 인스턴스
    """
    return LLMAgent(
        workflow_manager=dependencies['workflow_manager'],
        decision_engine=dependencies['decision_engine'],
        context_manager=dependencies['context_manager'],
        llm_client=dependencies['llm_client'],
        prompt_crafter=dependencies['prompt_crafter'],
        data_loader=dependencies['data_loader'],
        # Enhanced RAG 시스템 컴포넌트들 추가
        business_retriever=dependencies['business_retriever'],
        schema_retriever=dependencies['schema_retriever'],
        rag_manager=dependencies['rag_manager'],
        analysis_recommender=dependencies['analysis_recommender'],
        safe_code_executor=dependencies['safe_code_executor'],
        report_generator=dependencies['report_generator']
    ) 