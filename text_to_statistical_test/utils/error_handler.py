"""
애플리케이션 전역에서 사용될 커스텀 예외 클래스를 정의합니다.
"""
from enum import IntEnum

class ErrorCode(IntEnum):
    """애플리케이션 전역 에러 코드"""
    # 일반 오류
    UNKNOWN_ERROR = 1000
    VALIDATION_ERROR = 1001

    # LLM 관련 오류 (1100대)
    LLM_API_ERROR = 1100
    LLM_TIMEOUT_ERROR = 1101
    LLM_RESPONSE_ERROR = 1102
    MODEL_NOT_DEFINED = 1103
    
    # 파싱 관련 오류 (1200대)
    PARSING_ERROR = 1200

class BaseAppException(Exception):
    """모든 커스텀 예외의 기본 클래스"""
    def __init__(self, message, error_code=None):
        super().__init__(message)
        self.error_code = error_code

class LLMException(BaseAppException):
    """LLM 관련 오류"""
    pass

class ParsingException(BaseAppException):
    """LLM 응답 파싱 관련 오류"""
    pass

class PromptException(BaseAppException):
    """프롬프트 로딩 또는 포맷팅 관련 오류"""
    pass

class RAGException(BaseAppException):
    """RAG 서비스 관련 오류 (초기화, 검색, 수집 등)"""
    pass

class StatisticalException(BaseAppException):
    """통계 분석 도구 실행 관련 오류"""
    pass

class VisualizationException(BaseAppException):
    """시각화 생성 관련 오류"""
    pass

class DataElementException(BaseAppException):
    """데이터 로딩 또는 검증 관련 오류"""
    pass

class ErrorHandler:
    """
    간단한 오류 처리 및 로깅을 위한 클래스.
    [TODO] 향후 Sentry 등 외부 모니터링 서비스와 연동 가능.
    """
    def __init__(self, logger):
        self.logger = logger

    def handle(self, error: Exception, message: str = "오류 발생"):
        """오류를 로깅합니다."""
        self.logger.error(f"{message}: {error.__class__.__name__} - {error}")
        # [TODO] 필요시 오류를 다시 발생시키거나 다른 조치를 취할 수 있음. 