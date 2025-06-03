# src/config_loader.py

import yaml
import os
import logging
import logging.config
from typing import Dict, Any
from dotenv import load_dotenv # python-dotenv 라이브러리 import

# 프로젝트 루트 디렉토리 설정 (src 폴더의 부모 디렉토리)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# .env 파일 경로 (프로젝트 루트)
DOTENV_PATH = os.path.join(PROJECT_ROOT, '.env')

# 설정 파일 경로
SETTINGS_FILE_PATH = os.path.join(PROJECT_ROOT, 'config', 'settings.yaml')
LOGGING_CONFIG_FILE_PATH = os.path.join(PROJECT_ROOT, 'config', 'logging_config.yaml')

# 로그 디렉토리 경로 설정 및 생성
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

# 전역 설정 객체
APP_SETTINGS: Dict[str, Any] = {}


def load_dotenv_file() -> bool:
    """
    프로젝트 루트의 .env 파일에서 환경 변수를 로드합니다.
    .env 파일이 없어도 오류를 발생시키지 않고, 로드 성공 여부를 반환합니다.
    """
    if os.path.exists(DOTENV_PATH):
        loaded = load_dotenv(DOTENV_PATH, verbose=True) # verbose=True는 로드된 파일 경로를 출력
        if loaded:
            return True
        else:
            return False # 파일은 있지만 변수가 없을 수도 있음
    else:
        return False


def load_yaml_config(file_path: str, config_name: str) -> Dict[str, Any]:
    """
    지정된 YAML 파일을 로드합니다.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{config_name} file not found at: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        return config_data if config_data else {}
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing {config_name} file ({file_path}): {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while loading {config_name} file ({file_path}): {e}")


def setup_logging(logging_settings: Dict[str, Any]) -> None:
    """
    제공된 설정을 사용하여 로깅 환경을 구성합니다.
    """
    try:
        # 로그 파일 경로가 logging_config.yaml에 상대 경로로 지정되어 있다면,
        # 프로젝트 루트 기준으로 절대 경로로 변환
        if 'handlers' in logging_settings and 'file' in logging_settings['handlers']:
            log_file_path = logging_settings['handlers']['file'].get('filename')
            if log_file_path and not os.path.isabs(log_file_path):
                absolute_log_file_path = os.path.join(PROJECT_ROOT, log_file_path)
                log_file_dir = os.path.dirname(absolute_log_file_path)
                if not os.path.exists(log_file_dir):
                    try:
                        os.makedirs(log_file_dir, exist_ok=True) # exist_ok=True 로 동시성 문제 일부 완화
                    except OSError as e:
                        print(f"Warning: Could not create directory for log file {absolute_log_file_path}. Error: {e}")
                logging_settings['handlers']['file']['filename'] = absolute_log_file_path
        
        logging.config.dictConfig(logging_settings)
        # 초기 로깅 설정이 완료된 후 첫 로그 메시지를 남길 수 있습니다.
        # logger = logging.getLogger(__name__) # 여기서 로거를 가져오면 config_loader 로거가 됨
        # logger.info("Logging configured successfully.") # 이 메시지는 아래 load_app_config 에서 처리
    except Exception as e:
        # 로깅 설정 실패 시, print를 사용하고 기본 로깅으로 대체
        print(f"Error during logging setup from config: {e}. Using basic logging.")
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_app_config() -> None:
    """
    애플리케이션의 모든 설정을 로드하고 초기화합니다.
    1. .env 파일 로드 (환경 변수 설정)
    2. logging_config.yaml 로드 및 로깅 설정
    3. settings.yaml 로드 및 전역 APP_SETTINGS에 저장
    이 함수는 애플리케이션 시작 시 한 번만 호출되어야 합니다.
    """
    global APP_SETTINGS # 전역 변수 APP_SETTINGS 수정 명시

    # 1. .env 파일 로드
    env_loaded = load_dotenv_file()
    # .env 로드 후 바로 로깅 설정을 해야 환경변수를 로깅 설정에서도 사용할 수 있음 (필요시)

    # 2. 로깅 설정
    try:
        logging_settings = load_yaml_config(LOGGING_CONFIG_FILE_PATH, "Logging config")
        setup_logging(logging_settings)
    except Exception as e:
        # 로깅 설정 파일 로드 또는 적용 실패 시 기본 로깅 사용
        print(f"Failed to load or apply logging configuration: {e}. Using basic logging.")
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        # 이 시점에서는 아직 전용 로거가 없을 수 있으므로 print 사용
    
    logger = logging.getLogger(__name__) # 로깅 설정 후 로거 가져오기
    if env_loaded:
        logger.info(f"Loaded environment variables from: {DOTENV_PATH}")
    else:
        logger.info(f".env file not found or empty at: {DOTENV_PATH}. Relying on pre-set environment variables.")

    # 3. 주요 애플리케이션 설정 로드
    try:
        settings = load_yaml_config(SETTINGS_FILE_PATH, "Main application settings")
        APP_SETTINGS.update(settings) # 전역 설정 객체 업데이트
        logger.info("Application settings loaded successfully.")
        logger.debug(f"Loaded settings: {APP_SETTINGS}")
    except Exception as e:
        logger.critical(f"Failed to load main application settings: {e}", exc_info=True)
        raise # 설정 로드 실패는 심각한 문제이므로 예외를 다시 발생시켜 애플리케이션 중단 유도


# 다른 모듈에서 APP_SETTINGS를 import 하여 사용 가능
# 예: from src.config_loader import APP_SETTINGS
# my_llm_model = APP_SETTINGS.get('llm', {}).get('model_name')

if __name__ == '__main__':
    # 애플리케이션 설정 로드 및 로깅 초기화
    try:
        load_app_config()
        main_logger = logging.getLogger(__name__) # 이 파일의 로거
        
        main_logger.info("Config loader test: Application settings and logging initialized.")
        main_logger.debug(f"LLM Provider from settings: {APP_SETTINGS.get('llm', {}).get('provider')}")
        
        # .env 파일에서 로드된 환경 변수 사용 예시 (API 키 직접 로깅은 피해야 함)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            main_logger.info("OPENAI_API_KEY is set in environment.")
            # main_logger.debug(f"OpenAI API Key: {openai_api_key[:5]}...") # 실제 키 일부만 로깅 (주의)
        else:
            main_logger.warning("OPENAI_API_KEY is not set in environment. LLM services might fail.")

    except Exception as e:
        # load_app_config에서 이미 로깅이 설정되었거나 기본 로깅이 활성화됨
        # logging.getLogger(__name__).critical(f"Error in config_loader test: {e}", exc_info=True)
        # 만약 load_app_config 자체가 실패하면 로깅이 안될 수 있으므로 print도 고려
        print(f"Critical error in config_loader test: {e}")