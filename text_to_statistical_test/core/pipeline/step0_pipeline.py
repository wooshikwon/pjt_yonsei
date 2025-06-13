"""
파이프라인 단계의 추상 기반 클래스(Abstract Base Class)를 정의합니다.
"""

from core.pipeline.app_context import AppContext

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
    # [수정] 반환 타입을 명확한 AppContext로 변경
    async def run(self, context: AppContext) -> AppContext:
        """
        각 단계의 핵심 로직을 실행합니다.
        
        Args:
            context: 워크플로우의 현재 상태를 담고 있는 AppContext 객체.

        Returns:
            업데이트된 AppContext 객체.
        """
        pass

    def __str__(self) -> str:
        return f"PipelineStep(name='{self.step_name}')" 