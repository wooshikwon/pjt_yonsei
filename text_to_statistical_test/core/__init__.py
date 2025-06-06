"""
Core Package

Text-to-Statistical-Test 시스템의 핵심 로직을 담고 있는 패키지
- pipeline: 8단계 워크플로우 모듈
- rag: RAG(Retrieval Augmented Generation) 시스템
- agent: Agentic LLM 시스템
- workflow: 워크플로우 오케스트레이션
- reporting: 결과 보고 및 시각화
"""

from .pipeline import *
from . import rag
from . import agent
from . import workflow
from . import reporting

__version__ = "1.0.0"
__author__ = "Text-to-Statistical-Test Team"

__all__ = [
    'pipeline',
    'rag', 
    'agent',
    'workflow',
    'reporting'
] 