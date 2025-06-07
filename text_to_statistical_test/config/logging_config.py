"""
Logging Configuration

애플리케이션의 로깅 시스템을 설정합니다.
- 콘솔 출력과 파일 로깅을 모두 처리합니다.
- 설정 값은 config.settings 모듈에서 가져와 일관성을 유지합니다.
"""

import logging
import sys
from datetime import datetime

# 설정 모듈에서 필요한 설정 값을 가져옵니다.
from config.settings import get_settings

def setup_logging():
    """
    애플리케이션의 로깅 시스템을 초기화하고 구성합니다.
    
    이 함수는 두 개의 핸들러를 설정합니다:
    1. 콘솔 핸들러: INFO 레벨 이상의 로그를 표준 출력으로 보냅니다.
    2. 파일 핸들러: DEBUG 레벨 이상의 모든 로그를 타임스탬프가 찍힌
       로그 파일에 저장합니다.
    """
    settings = get_settings()

    # 루트 로거의 기본 레벨을 DEBUG로 설정하여 모든 레벨의 로그를 처리할 수 있게 합니다.
    # 각 핸들러에서 실제 출력 레벨을 제어합니다.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # 기존에 연결된 핸들러가 있다면 모두 제거하여 중복 로깅을 방지합니다.
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # --- 콘솔 핸들러 설정 ---
    # 사용자가 핵심 진행 상황만 볼 수 있도록 콘솔 레벨을 WARNING으로 설정합니다.
    # 중요한 사용자 안내는 logger.info 대신 print()를 사용하도록 코드를 수정합니다.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING) 
    # 에러/경고 메시지는 원인을 파악하기 쉽도록 포맷을 추가합니다.
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # --- 파일 핸들러 설정 ---
    # 로그 파일은 'logs/app_YYYYMMDD.log' 형식으로 생성됩니다.
    log_filename = f"app_{datetime.now().strftime('%Y%m%d')}.log"
    log_filepath = settings.paths.logs_dir / log_filename

    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    # 파일에는 항상 DEBUG 레벨 이상의 모든 상세 정보를 기록합니다.
    file_handler.setLevel(logging.DEBUG)
    # 파일에는 타임스탬프, 로그 레벨, 모듈명 등 상세 정보를 기록합니다.
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # 서드파티 라이브러리의 과도한 로깅을 방지합니다.
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    logging.info("로깅 시스템이 성공적으로 초기화되었습니다.")
    logging.debug(f"콘솔 로그 레벨: {settings.app.log_level}, 로그 파일 경로: {log_filepath}") 