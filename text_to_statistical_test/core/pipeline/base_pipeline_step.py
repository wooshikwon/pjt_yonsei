"""
Base Pipeline Step Class for Text-to-Statistical-Test System

모든 파이프라인 단계가 상속받을 추상 기본 클래스
8단계 워크플로우의 표준화된 인터페이스 제공
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
from datetime import datetime


class BasePipelineStep(ABC):
    """
    파이프라인 단계의 기본 추상 클래스
    모든 단계는 명확한 입력과 출력을 가져야 함
    """
    
    def __init__(self, step_name: str, step_number: int):
        """
        Args:
            step_name: 단계 이름
            step_number: 단계 번호 (1-8)
        """
        self.step_name = step_name
        self.step_number = step_number
        self.logger = logging.getLogger(f"{__name__}.{step_name}")
        self.execution_start_time = None
        self.execution_end_time = None
        self.step_results = {}
        
    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        단계 실행 - 각 단계에서 반드시 구현해야 함
        
        Args:
            input_data: 이전 단계에서 전달받은 데이터
            
        Returns:
            Dict[str, Any]: 다음 단계로 전달할 결과 데이터
        """
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        입력 데이터 유효성 검증
        
        Args:
            input_data: 검증할 입력 데이터
            
        Returns:
            bool: 유효성 검증 결과
        """
        pass
    
    @abstractmethod
    def get_expected_output_schema(self) -> Dict[str, Any]:
        """
        예상 출력 스키마 반환
        
        Returns:
            Dict[str, Any]: 출력 데이터 스키마
        """
        pass
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        단계 실행 래퍼 - 로깅, 시간 측정, 오류 처리 포함
        
        Args:
            input_data: 입력 데이터
            
        Returns:
            Dict[str, Any]: 실행 결과
        """
        try:
            self.execution_start_time = datetime.now()
            self.logger.info(f"Step {self.step_number} ({self.step_name}) 시작")
            
            # 입력 유효성 검증
            if not self.validate_input(input_data):
                raise ValueError(f"Invalid input data for step {self.step_number}")
            
            # 단계 실행
            result = self.execute(input_data)
            
            # 결과 저장
            self.step_results = result
            self.execution_end_time = datetime.now()
            
            execution_time = (self.execution_end_time - self.execution_start_time).total_seconds()
            self.logger.info(f"Step {self.step_number} ({self.step_name}) 완료 - 실행시간: {execution_time:.2f}초")
            
            # 메타데이터 추가
            result['_meta'] = {
                'step_number': self.step_number,
                'step_name': self.step_name,
                'execution_time': execution_time,
                'timestamp': self.execution_end_time.isoformat(),
                'success': True
            }
            
            return result
            
        except Exception as e:
            self.execution_end_time = datetime.now()
            error_msg = f"Step {self.step_number} ({self.step_name}) 실행 중 오류: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return {
                'error': True,
                'error_message': str(e),
                'step_number': self.step_number,
                'step_name': self.step_name,
                '_meta': {
                    'step_number': self.step_number,
                    'step_name': self.step_name,
                    'execution_time': 0,
                    'timestamp': self.execution_end_time.isoformat() if self.execution_end_time else None,
                    'success': False
                }
            }
    
    def get_step_info(self) -> Dict[str, Any]:
        """
        단계 정보 반환
        
        Returns:
            Dict[str, Any]: 단계 메타정보
        """
        return {
            'step_number': self.step_number,
            'step_name': self.step_name,
            'execution_start_time': self.execution_start_time.isoformat() if self.execution_start_time else None,
            'execution_end_time': self.execution_end_time.isoformat() if self.execution_end_time else None,
            'has_results': bool(self.step_results)
        }
    
    def reset(self):
        """단계 상태 초기화"""
        self.execution_start_time = None
        self.execution_end_time = None
        self.step_results = {}
        self.logger.debug(f"Step {self.step_number} ({self.step_name}) 상태 초기화")


class PipelineStepRegistry:
    """
    파이프라인 단계 등록 및 관리 클래스
    """
    
    _steps = {}
    
    @classmethod
    def register_step(cls, step_number: int, step_class: type):
        """
        파이프라인 단계 등록
        
        Args:
            step_number: 단계 번호
            step_class: 단계 클래스
        """
        cls._steps[step_number] = step_class
    
    @classmethod
    def get_step(cls, step_number: int) -> Optional[type]:
        """
        등록된 단계 클래스 반환
        
        Args:
            step_number: 단계 번호
            
        Returns:
            Optional[type]: 단계 클래스
        """
        return cls._steps.get(step_number)
    
    @classmethod
    def get_all_steps(cls) -> Dict[int, type]:
        """
        모든 등록된 단계 반환
        
        Returns:
            Dict[int, type]: 단계 번호와 클래스 매핑
        """
        return cls._steps.copy()
    
    @classmethod
    def validate_pipeline(cls) -> bool:
        """
        파이프라인 완성도 검증
        
        Returns:
            bool: 8단계 모두 등록되었는지 여부
        """
        return set(cls._steps.keys()) == set(range(1, 9)) 