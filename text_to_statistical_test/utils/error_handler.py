"""
오류 처리 유틸리티

표준화된 오류 처리 및 예외 정의
"""

import logging
import traceback
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ErrorCode(Enum):
    """
    표준화된 오류 코드
    """
    # 일반 오류
    UNKNOWN_ERROR = "E001"
    VALIDATION_ERROR = "E002"
    CONFIGURATION_ERROR = "E003"
    
    # 파일 관련 오류
    FILE_NOT_FOUND = "E101"
    FILE_READ_ERROR = "E102"
    FILE_WRITE_ERROR = "E103"
    FILE_FORMAT_ERROR = "E104"
    FILE_SIZE_ERROR = "E105"
    
    # 데이터 관련 오류
    DATA_LOADING_ERROR = "E201"
    DATA_VALIDATION_ERROR = "E202"
    DATA_PROCESSING_ERROR = "E203"
    DATA_EMPTY_ERROR = "E204"
    DATA_TYPE_ERROR = "E205"
    
    # 통계 분석 오류
    STATISTICAL_TEST_ERROR = "E301"
    ASSUMPTION_VIOLATION = "E302"
    INSUFFICIENT_DATA = "E303"
    INVALID_PARAMETER = "E304"
    
    # LLM 관련 오류
    LLM_API_ERROR = "E401"
    LLM_RESPONSE_ERROR = "E402"
    LLM_TIMEOUT_ERROR = "E403"
    LLM_QUOTA_ERROR = "E404"
    
    # RAG 관련 오류
    RAG_QUERY_ERROR = "E501"
    RAG_INDEX_ERROR = "E502"
    RAG_KNOWLEDGE_ERROR = "E503"
    
    # 워크플로우 오류
    WORKFLOW_ERROR = "E601"
    PIPELINE_ERROR = "E602"
    STATE_ERROR = "E603"

class CustomException(Exception):
    """
    커스텀 예외 기본 클래스
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.original_exception = original_exception
        self.timestamp = datetime.now()

class ValidationException(CustomException):
    """검증 관련 예외"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCode.VALIDATION_ERROR, details)

class DataException(CustomException):
    """데이터 관련 예외"""
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.DATA_PROCESSING_ERROR, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)

class DataProcessingException(CustomException):
    """데이터 처리 관련 예외"""
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.DATA_PROCESSING_ERROR, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)

class StatisticalException(CustomException):
    """통계 분석 관련 예외"""
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.STATISTICAL_TEST_ERROR, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)

class LLMException(CustomException):
    """LLM 관련 예외"""
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.LLM_API_ERROR, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)

class PromptException(CustomException):
    """프롬프트 관련 예외"""
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.LLM_RESPONSE_ERROR, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)

class ParsingException(CustomException):
    """파싱 관련 예외"""
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.LLM_RESPONSE_ERROR, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)

class StatisticsException(CustomException):
    """통계 관련 예외"""
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.STATISTICAL_TEST_ERROR, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)

class RAGException(CustomException):
    """RAG 관련 예외"""
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.RAG_QUERY_ERROR, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)

# RAGError는 RAGException의 별칭
RAGError = RAGException

class WorkflowException(CustomException):
    """워크플로우 관련 예외"""
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.WORKFLOW_ERROR, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)

# PipelineError는 WorkflowException의 별칭  
PipelineError = WorkflowException

class ErrorHandler:
    """
    통합 오류 처리 클래스
    """
    
    def __init__(self):
        self.error_log = []
        
    def handle_error(
        self, 
        error: Exception, 
        context: Optional[Dict[str, Any]] = None,
        default_return: Any = None
    ) -> Dict[str, Any]:
        """
        에러 처리 및 로깅
        
        Args:
            error: 발생한 예외
            context: 에러 컨텍스트
            default_return: 기본 반환값 (오류 시 반환할 값)
            
        Returns:
            Dict[str, Any]: 에러 정보
        """
        try:
            # 에러 정보 추출
            error_info = {
                'error_type': type(error).__name__,
                'message': str(error),
                'timestamp': datetime.now().isoformat(),
                'context': context or {},
                'default_return': default_return
            }
            
            # 스택 트레이스 추가 (개발 모드에서만)
            if logger.isEnabledFor(logging.DEBUG):
                error_info['traceback'] = traceback.format_exc()
            
            # 에러 분류
            error_category = self._categorize_error(error)
            error_info['category'] = error_category
            
            # 중요도 결정
            severity = self._determine_severity(error, error_category)
            error_info['severity'] = severity
            
            # 로깅
            self._log_error(error_info)
            
            # 통계 업데이트
            self._update_statistics(error_category, severity)
            
            # 에러 히스토리 저장
            if self.error_log:
                self.error_log.append(error_info)
            
            # 기본 반환값이 설정된 경우 바로 반환
            if default_return is not None:
                return default_return
            
            return error_info
            
        except Exception as handling_error:
            # 에러 처리 중 에러 발생
            fallback_info = {
                'error_type': 'ErrorHandlingFailed',
                'message': f'에러 처리 실패: {str(handling_error)}',
                'original_error': str(error),
                'timestamp': datetime.now().isoformat(),
                'default_return': default_return
            }
            
            # 기본 로거로 로깅
            logger.error(f"에러 처리 실패: {fallback_info}")
            
            # 기본 반환값이 설정된 경우 반환
            if default_return is not None:
                return default_return
                
            return fallback_info
    
    def _log_error(self, error_info: Dict[str, Any]):
        """
        오류 정보를 로그에 기록
        """
        try:
            # 심각도에 따른 로그 레벨 결정
            error_code = error_info.get('error_code')
            if error_code and error_code.startswith('E4'):  # LLM 관련 오류
                log_level = logging.WARNING
            elif error_code and error_code.startswith('E3'):  # 통계 분석 오류
                log_level = logging.ERROR
            else:
                log_level = logging.ERROR
            
            # 로그 메시지 구성
            log_message = (
                f"오류 발생 - 코드: {error_info.get('error_code', 'N/A')}, "
                f"타입: {error_info['error_type']}, "
                f"메시지: {error_info['message']}"
            )
            
            # 컨텍스트가 있으면 추가
            if error_info['context']:
                log_message += f", 컨텍스트: {json.dumps(error_info['context'], ensure_ascii=False)}"
            
            logger.log(log_level, log_message)
            
            # 디버그 모드에서는 traceback도 로깅
            if error_info.get('traceback'):
                logger.debug(f"Traceback: {error_info['traceback']}")
                
        except Exception as e:
            # 로깅 중 오류 발생 시 기본 로깅으로 대체
            logger.error(f"오류 로깅 중 문제 발생: {str(e)}")
    
    def _get_recovery_suggestions(self, error_code: ErrorCode) -> List[str]:
        """
        오류 코드별 복구 제안 반환
        """
        suggestions = {
            ErrorCode.FILE_NOT_FOUND: [
                "파일 경로를 다시 확인해주세요.",
                "파일이 존재하는지 확인해주세요.",
                "파일 권한을 확인해주세요."
            ],
            ErrorCode.FILE_FORMAT_ERROR: [
                "지원하는 파일 형식(CSV, Excel, JSON)인지 확인해주세요.",
                "파일이 손상되지 않았는지 확인해주세요.",
                "파일 인코딩을 확인해주세요."
            ],
            ErrorCode.DATA_EMPTY_ERROR: [
                "데이터가 포함된 파일을 선택해주세요.",
                "빈 행이나 열을 제거해주세요.",
                "데이터 형식을 확인해주세요."
            ],
            ErrorCode.INSUFFICIENT_DATA: [
                "더 많은 데이터가 필요합니다.",
                "다른 통계 검정 방법을 시도해보세요.",
                "데이터 수집을 늘려보세요."
            ],
            ErrorCode.ASSUMPTION_VIOLATION: [
                "비모수 검정을 고려해보세요.",
                "데이터 변환을 시도해보세요.",
                "이상치를 확인하고 제거해보세요."
            ],
            ErrorCode.LLM_API_ERROR: [
                "잠시 후 다시 시도해주세요.",
                "API 키가 유효한지 확인해주세요.",
                "네트워크 연결을 확인해주세요."
            ],
            ErrorCode.LLM_QUOTA_ERROR: [
                "API 사용량 한도를 확인해주세요.",
                "다른 모델을 사용해보세요.",
                "잠시 후 다시 시도해주세요."
            ]
        }
        
        return suggestions.get(error_code, [
            "문제가 지속되면 시스템 관리자에게 문의해주세요.",
            "입력 데이터를 다시 확인해주세요."
        ])
    
    def create_user_friendly_message(self, error: Exception) -> str:
        """
        사용자 친화적인 오류 메시지 생성
        """
        if isinstance(error, ValidationException):
            return f"입력 검증 오류: {error.message}"
        elif isinstance(error, DataException):
            return f"데이터 처리 오류: {error.message}"
        elif isinstance(error, StatisticalException):
            return f"통계 분석 오류: {error.message}"
        elif isinstance(error, LLMException):
            return "AI 서비스 연결에 문제가 있습니다. 잠시 후 다시 시도해주세요."
        elif isinstance(error, RAGException):
            return "지식 검색 중 문제가 발생했습니다."
        elif isinstance(error, WorkflowException):
            return f"작업 처리 중 오류가 발생했습니다: {error.message}"
        else:
            return "예상치 못한 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        오류 로그 요약 반환
        """
        if not self.error_log:
            return {
                'total_errors': 0,
                'error_types': {},
                'recent_errors': []
            }
        
        error_types = {}
        for error in self.error_log:
            error_type = error.get('error_code', error['error_type'])
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': len(self.error_log),
            'error_types': error_types,
            'recent_errors': self.error_log[-5:]  # 최근 5개 오류
        }
    
    def clear_error_log(self):
        """
        오류 로그 초기화
        """
        self.error_log.clear()
        logger.info("오류 로그가 초기화되었습니다.")

    def _categorize_error(self, error: Exception) -> str:
        """에러 분류"""
        error_type = type(error).__name__
        
        if 'FileNotFound' in error_type or 'Path' in error_type:
            return 'file_error'
        elif 'Permission' in error_type or 'Access' in error_type:
            return 'permission_error'
        elif 'Memory' in error_type or 'Resource' in error_type:
            return 'resource_error'
        elif 'Network' in error_type or 'Connection' in error_type:
            return 'network_error'
        elif 'Value' in error_type or 'Type' in error_type:
            return 'validation_error'
        elif 'Import' in error_type or 'Module' in error_type:
            return 'dependency_error'
        else:
            return 'runtime_error'
    
    def _determine_severity(self, error: Exception, category: str) -> str:
        """에러 심각도 결정"""
        if category in ['permission_error', 'resource_error', 'dependency_error']:
            return 'critical'
        elif category in ['file_error', 'network_error']:
            return 'high'
        elif category in ['validation_error']:
            return 'medium'
        else:
            return 'low'
    
    def _update_statistics(self, category: str, severity: str):
        """에러 통계 업데이트"""
        try:
            if not hasattr(self, 'error_stats'):
                self.error_stats = {
                    'total_errors': 0,
                    'by_category': {},
                    'by_severity': {}
                }
            
            self.error_stats['total_errors'] += 1
            
            if category not in self.error_stats['by_category']:
                self.error_stats['by_category'][category] = 0
            self.error_stats['by_category'][category] += 1
            
            if severity not in self.error_stats['by_severity']:
                self.error_stats['by_severity'][severity] = 0
            self.error_stats['by_severity'][severity] += 1
            
        except Exception as e:
            logger.error(f"에러 통계 업데이트 실패: {e}")

# 전역 인스턴스
error_handler = ErrorHandler()

# 편의 함수들
def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None, user_message: Optional[str] = None) -> Dict[str, Any]:
    """오류 처리 편의 함수"""
    return error_handler.handle_error(error, context, user_message)

def create_user_message(error: Exception) -> str:
    """사용자 친화적 메시지 생성 편의 함수"""
    return error_handler.create_user_friendly_message(error) 