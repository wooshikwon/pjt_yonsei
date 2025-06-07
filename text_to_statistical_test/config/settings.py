"""
Application Settings

프로젝트 전반에서 사용되는 핵심 설정 값을 정의합니다.
이 파일은 하드코딩된 경로, 기본값 등을 중앙에서 관리하여
일관성을 유지하고 변경을 용이하게 합니다.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# .env 파일 로딩은 main.py에서 처리하므로 여기서는 os.getenv만 사용합니다.

# 프로젝트의 루트 디렉토리를 기준으로 경로를 설정합니다.
PROJECT_ROOT = Path(__file__).parent.parent

@dataclass
class PathSettings:
    """애플리케이션에서 사용하는 주요 경로들을 정의합니다."""
    project_root: Path = PROJECT_ROOT
    resources_dir: Path = PROJECT_ROOT / "resources"
    input_data_dir: Path = PROJECT_ROOT / "input_data"
    output_data_dir: Path = PROJECT_ROOT / "output_data"
    reports_dir: Path = output_data_dir / "reports"
    visualizations_dir: Path = output_data_dir / "visualizations"
    logs_dir: Path = PROJECT_ROOT / "logs"

@dataclass
class LLMSettings:
    """LLM API와 관련된 설정을 정의합니다."""
    # .env 파일에서 키를 로드합니다. 키가 없으면 None이 됩니다.
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    default_model: str = "gpt-4-turbo"
    temperature: float = 0.2
    max_tokens: int = 4096

@dataclass
class AppSettings:
    """애플리케이션의 일반 설정을 정의합니다."""
    # .env 파일이나 환경변수에서 값을 가져옵니다. 없으면 기본값이 사용됩니다.
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    supported_file_formats: List[str] = field(default_factory=lambda: ['.csv', '.xlsx', '.xls'])
    # RAG 서비스 관련 설정 추가
    knowledge_base_dir: Path = PROJECT_ROOT / "resources" / "knowledge_base"
    rag_storage_path: Path = PROJECT_ROOT / "output_data" / "rag_storage"

# 전체 설정을 통합하는 컨테이너 클래스
@dataclass
class Settings:
    paths: PathSettings
    llm: LLMSettings
    app: AppSettings

def get_settings() -> Settings:
    """
    모든 설정 클래스를 포함하는 단일 Settings 객체를 반환합니다.
    """
    return Settings(
        paths=PathSettings(),
        llm=LLMSettings(),
        app=AppSettings()
    )

# 모듈이 임포트될 때 디렉토리 생성 로직을 한 번 실행합니다.
def ensure_directories_exist():
    """
    애플리케이션 실행에 필요한 출력 디렉토리들이 존재하는지 확인하고,
    없으면 생성합니다.
    """
    settings = get_settings()
    dirs_to_create = [
        settings.paths.reports_dir,
        settings.paths.visualizations_dir,
        settings.paths.logs_dir,
        settings.app.rag_storage_path,
    ]
    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)

ensure_directories_exist() 