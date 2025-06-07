# 파일명: core/pipeline/base_pipeline_step.py
"""
파이프라인 단계의 추상 기반 클래스(Abstract Base Class)를 정의합니다.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Coroutine

class PipelineStep(ABC):
    """
    워크플로우를 구성하는 각 단계의 추상 기본 클래스.
    모든 파이프라인 단계는 이 클래스를 상속받아야 합니다.
    """
    def __init__(self, step_name: str):
        """
        Args:
            step_name (str): 파이프라인 단계의 이름 (로깅 및 식별용).
        """
        if not step_name:
            raise ValueError("파이프라인 단계의 이름은 비어 있을 수 없습니다.")
        self._step_name = step_name
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"'{self._step_name}' 단계가 초기화되었습니다.")

    @property
    def step_name(self) -> str:
        """이 단계의 이름을 반환합니다."""
        return self._step_name

    @abstractmethod
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any] | Coroutine[Any, Any, Dict[str, Any]]:
        """
        각 단계의 핵심 로직을 실행합니다.
        
        Args:
            context: 워크플로우의 현재 상태를 담고 있는 딕셔너리.
                     이전 단계들로부터 전달된 모든 데이터를 포함합니다.

        Returns:
            context를 업데이트할 새로운 데이터가 담긴 딕셔너리.
            이 딕셔너리는 기존 context에 병합됩니다.
        """
        pass 

    def __str__(self) -> str:
        return f"PipelineStep(name='{self.step_name}')" 