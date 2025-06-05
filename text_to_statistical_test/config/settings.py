"""
애플리케이션 전반의 설정값 관리

환경 변수에서 로드하거나 기본값을 제공하는 설정 중앙화 모듈
"""

import os
from pathlib import Path


# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent

# LLM 관련 설정
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# OpenAI API 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 워크플로우 설정
WORKFLOW_FILE_PATH = str(PROJECT_ROOT / "resources" / "workflow_graph.json")

# RAG 시스템 설정
CODE_SNIPPETS_DIR = str(PROJECT_ROOT / "resources" / "code_snippets")
RAG_INDEX_PATH = str(PROJECT_ROOT / "resources" / "rag_index" / "code_snippets.index")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-ada-002")

# 프롬프트 설정
PROMPT_TEMPLATES_DIR = str(PROJECT_ROOT / "llm_services" / "prompts")

# 데이터 경로 설정
INPUT_DATA_DEFAULT_DIR = str(PROJECT_ROOT / "input_data")
OUTPUT_RESULTS_DIR = str(PROJECT_ROOT / "output_results")

# 로깅 설정
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# 컨텍스트 관리 설정
MAX_HISTORY_ITEMS = int(os.getenv("MAX_HISTORY_ITEMS", "20"))
SUMMARIZATION_TRIGGER_COUNT = int(os.getenv("SUMMARIZATION_TRIGGER_COUNT", "10"))
CONTEXT_TOKEN_LIMIT = int(os.getenv("CONTEXT_TOKEN_LIMIT", "3000"))

# 코드 실행 설정
CODE_EXECUTION_TIMEOUT = int(os.getenv("CODE_EXECUTION_TIMEOUT", "30"))
SAFE_CODE_EXECUTION = os.getenv("SAFE_CODE_EXECUTION", "true").lower() == "true"

# 보고서 설정
REPORT_FORMAT = os.getenv("REPORT_FORMAT", "md")  # "md", "html", "pdf"

# 개발/프로덕션 모드
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# 필수 디렉토리 생성 함수
def ensure_directories():
    """필요한 디렉토리들을 생성합니다."""
    directories_to_create = [
        INPUT_DATA_DEFAULT_DIR,
        OUTPUT_RESULTS_DIR,
        CODE_SNIPPETS_DIR,
        "logs",  # 로그 디렉토리
        "config",
        "llm_services/prompts",
        RAG_INDEX_PATH
    ]
    
    for directory in directories_to_create:
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"📁 디렉토리 생성: {directory}")
            except Exception as e:
                print(f"⚠️  디렉토리 생성 실패 ({directory}): {e}")

# 설정 검증 함수
def validate_settings():
    """환경 설정 검증"""
    errors = []
    
    # LLM 제공자 확인
    if LLM_PROVIDER.lower() != "openai":
        errors.append(f"지원하지 않는 LLM 제공자: {LLM_PROVIDER}. 'openai'만 지원됩니다.")
    
    # OpenAI API 키 확인
    if LLM_PROVIDER.lower() == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            errors.append("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    
    # 필수 디렉토리 경로 검증
    required_dirs = [
        WORKFLOW_FILE_PATH,
        CODE_SNIPPETS_DIR,
        INPUT_DATA_DEFAULT_DIR,
        OUTPUT_RESULTS_DIR
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(os.path.dirname(dir_path)):
            errors.append(f"필수 디렉토리가 존재하지 않습니다: {os.path.dirname(dir_path)}")
    
    if errors:
        raise ValueError("환경 설정 오류:\n" + "\n".join(f"  • {error}" for error in errors))

# 설정 요약 출력
def print_current_settings():
    """현재 설정값들을 출력합니다."""
    print("⚙️  현재 설정:")
    print(f"   LLM Provider: {LLM_PROVIDER}")
    print(f"   LLM Model: {LLM_MODEL_NAME}")
    print(f"   Input Data Dir: {INPUT_DATA_DEFAULT_DIR}")
    print(f"   Output Dir: {OUTPUT_RESULTS_DIR}")
    print(f"   Log Level: {LOG_LEVEL}")

def get_api_status():
    """API 키들의 상태를 확인합니다"""
    status = {}
    
    # OpenAI 상태 확인
    openai_key = os.getenv("OPENAI_API_KEY")
    status["openai"] = {
        "available": bool(openai_key),
        "key_preview": f"{openai_key[:10]}..." if openai_key else "없음"
    }
    
    return status 