"""
Core Package

Text-to-Statistical-Test 시스템의 핵심 로직을 담고 있는 '지휘자(Orchestrator)' 계층.
- workflow: 5단계 파이프라인의 실행을 총괄하는 Orchestrator.
- pipeline: 워크플로우를 구성하는 각 실행 단계(Step) 정의.
- agent: 자율적으로 통계 분석을 수행하는 Agent.
- reporting: 최종 결과 보고서를 생성하는 Builder.
"""

from . import agent
from . import pipeline
from . import reporting
from . import workflow

__version__ = "1.0.0"
__author__ = "Text-to-Statistical-Test Team"

__all__ = [
    'agent',
    'pipeline',
    'reporting',
    'workflow',
] 