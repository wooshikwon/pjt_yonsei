"""
Logging Configuration

로깅 레벨, 포맷, 핸들러 등 상세 설정
"""

import logging
import logging.config
from pathlib import Path
from datetime import datetime
import sys
import os

def setup_logging(
    log_level: str = "INFO",
    log_file: str = None,
    console_output: bool = True,
    structured_logging: bool = True
) -> None:
    """
    로깅 시스템 설정
    
    Args:
        log_level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 로그 파일 경로 (None이면 기본 경로 사용)
        console_output: 콘솔 출력 여부
        structured_logging: 구조화된 로깅 사용 여부
    """
    
    # 로그 디렉토리 생성
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # 기본 로그 파일 이름
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"app_{timestamp}.log"
    
    # 로그 포맷 설정
    if structured_logging:
        log_format = (
            '%(asctime)s | %(levelname)-8s | %(name)-20s | '
            '%(filename)s:%(lineno)d | %(funcName)s | %(message)s'
        )
        date_format = '%Y-%m-%d %H:%M:%S'
    else:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
    
    # 핸들러 설정
    handlers = []
    
    # 파일 핸들러 (상세한 로그)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # 파일에는 모든 로그 저장
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    handlers.append(file_handler)
    
    # 콘솔 핸들러 (사용자 친화적)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        
        # 콘솔은 WARNING 이상만 출력 (중요한 메시지만)
        console_level = logging.WARNING if not os.getenv('DEBUG', 'false').lower() == 'true' else logging.DEBUG
        console_handler.setLevel(console_level)
        
        # 콘솔용 간단한 포맷 (타임스탬프 제거)
        console_format = '%(message)s'  # 메시지만 출력
        console_handler.setFormatter(logging.Formatter(console_format))
        
        # 필터 추가 - 시스템 로그는 콘솔에서 제외
        class UserMessageFilter(logging.Filter):
            def filter(self, record):
                # 사용자에게 중요한 메시지만 콘솔에 출력
                important_loggers = ['__main__', 'core.pipeline']
                
                # 디버그 모드가 아니면 INFO 레벨 시스템 로그 제외
                if not os.getenv('DEBUG', 'false').lower() == 'true':
                    if record.levelno < logging.WARNING:
                        # 파이프라인 단계 진행 메시지는 허용
                        if any(logger in record.name for logger in important_loggers):
                            return 'Step' in record.getMessage() or '단계' in record.getMessage() or '❌' in record.getMessage() or '✅' in record.getMessage()
                        return False
                return True
        
        console_handler.addFilter(UserMessageFilter())
        handlers.append(console_handler)
    
    # 로깅 설정 적용
    logging.basicConfig(
        level=logging.DEBUG,  # 루트 레벨은 DEBUG로 설정
        handlers=handlers,
        force=True  # 기존 설정 덮어쓰기
    )
    
    # 특정 라이브러리 로그 레벨 조정 (콘솔/파일 모두)
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('requests').setLevel(logging.ERROR)
    logging.getLogger('httpx').setLevel(logging.ERROR)
    logging.getLogger('faiss').setLevel(logging.ERROR)
    logging.getLogger('utils.data_loader').setLevel(logging.ERROR)
    logging.getLogger('utils.global_cache').setLevel(logging.ERROR)
    logging.getLogger('core.workflow.state_manager').setLevel(logging.ERROR)
    logging.getLogger('core.workflow.orchestrator').setLevel(logging.WARNING)
    
    # 성공 메시지 (파일에만)
    logger = logging.getLogger(__name__)
    logger.debug(f"Logging initialized - Level: {log_level}, File: {log_file}")

def get_logger(name: str) -> logging.Logger:
    """
    로거 인스턴스 반환
    
    Args:
        name: 로거 이름 (보통 __name__ 사용)
        
    Returns:
        logging.Logger: 로거 인스턴스
    """
    return logging.getLogger(name)

def configure_pipeline_logging() -> None:
    """파이프라인 모듈들을 위한 특별 로깅 설정"""
    
    # 파이프라인 단계별 로거 설정
    pipeline_modules = [
        'core.pipeline.data_selection',
        'core.pipeline.user_request', 
        'core.pipeline.data_summary',
        'core.pipeline.analysis_proposal',
        'core.pipeline.user_selection',
        'core.pipeline.agent_analysis',
        'core.pipeline.agent_execution',
        'core.pipeline.report_generation'
    ]
    
    for module in pipeline_modules:
        logger = logging.getLogger(module)
        logger.setLevel(logging.INFO)

def configure_debug_logging() -> None:
    """디버그 모드를 위한 상세 로깅 설정"""
    
    # 모든 모듈을 DEBUG 레벨로 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # 핸들러들도 DEBUG 레벨로 변경
    for handler in root_logger.handlers:
        handler.setLevel(logging.DEBUG)
    
    logger = logging.getLogger(__name__)
    logger.debug("Debug logging enabled")

# 기본 로깅 설정 (애플리케이션 시작시 호출)
def init_default_logging():
    """기본 로깅 설정 초기화"""
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    debug_mode = os.getenv('DEBUG', 'false').lower() == 'true'
    
    setup_logging(
        log_level=log_level,
        console_output=True,
        structured_logging=True
    )
    
    configure_pipeline_logging()
    
    if debug_mode:
        configure_debug_logging()

# 모듈 임포트시 자동 초기화
if __name__ != '__main__':
    init_default_logging() 