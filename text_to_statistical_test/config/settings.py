"""
Application Settings

통합 설정 파일
- API 키 관리
- 기본 경로 설정
- LLM 모델 설정
- 데이터베이스 설정 등
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent

@dataclass
class LLMSettings:
    """LLM 관련 설정"""
    default_model: str = "gpt-4-turbo-preview"
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.1
    timeout: int = 60
    
    def __post_init__(self):
        """환경변수에서 API 키 로드"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

@dataclass  
class PathSettings:
    """경로 관련 설정"""
    project_root: Path = PROJECT_ROOT
    input_data_dir: Path = PROJECT_ROOT / "input_data"
    data_files_dir: Path = PROJECT_ROOT / "input_data" / "data_files"
    metadata_dir: Path = PROJECT_ROOT / "input_data" / "metadata"
    output_data_dir: Path = PROJECT_ROOT / "output_data"
    reports_dir: Path = PROJECT_ROOT / "output_data" / "reports"
    visualizations_dir: Path = PROJECT_ROOT / "output_data" / "visualizations"
    cache_dir: Path = PROJECT_ROOT / "output_data" / "analysis_cache"
    logs_dir: Path = PROJECT_ROOT / "logs"
    resources_dir: Path = PROJECT_ROOT / "resources"
    knowledge_base_dir: Path = PROJECT_ROOT / "resources" / "knowledge_base"

@dataclass
class RAGSettings:
    """RAG 시스템 설정"""
    vector_store_type: str = "faiss"  # faiss, chroma, lancedb 등
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_search_results: int = 10
    similarity_threshold: float = 0.7
    cache_ttl: int = 3600  # 1시간

@dataclass
class DatabaseSettings:
    """데이터베이스 설정 (필요시)"""
    db_type: str = "sqlite"
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "text_to_statistical_test"
    db_user: Optional[str] = None
    db_password: Optional[str] = None

@dataclass
class StatisticsSettings:
    """통계 분석 설정"""
    significance_level: float = 0.05
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    random_seed: int = 42
    max_categories: int = 20  # 범주형 변수 최대 카테고리 수

@dataclass
class ApplicationSettings:
    """애플리케이션 전체 설정"""
    debug: bool = False
    log_level: str = "INFO"
    max_file_size_mb: int = 100
    supported_file_formats: list = None
    
    def __post_init__(self):
        if self.supported_file_formats is None:
            self.supported_file_formats = ['.csv', '.xlsx', '.xls', '.json', '.parquet', '.tsv']

# 설정 인스턴스 생성
llm_settings = LLMSettings()
path_settings = PathSettings()
rag_settings = RAGSettings()
db_settings = DatabaseSettings()
stats_settings = StatisticsSettings()
app_settings = ApplicationSettings()

def get_settings() -> Dict[str, Any]:
    """모든 설정을 딕셔너리로 반환"""
    return {
        'llm': llm_settings,
        'paths': path_settings,
        'rag': rag_settings,
        'database': db_settings,
        'statistics': stats_settings,
        'application': app_settings
    }

def update_settings(settings_dict: Dict[str, Any]) -> None:
    """설정 업데이트"""
    global llm_settings, path_settings, rag_settings, db_settings, stats_settings, app_settings
    
    if 'llm' in settings_dict:
        for key, value in settings_dict['llm'].items():
            if hasattr(llm_settings, key):
                setattr(llm_settings, key, value)
    
    # 다른 설정들도 유사하게 처리...
    
def ensure_directories():
    """필요한 디렉토리들이 존재하는지 확인하고 생성"""
    directories = [
        path_settings.input_data_dir,
        path_settings.data_files_dir,
        path_settings.metadata_dir,
        path_settings.output_data_dir,
        path_settings.reports_dir,
        path_settings.visualizations_dir,
        path_settings.cache_dir,
        path_settings.logs_dir
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# 시작시 디렉토리 생성
ensure_directories() 