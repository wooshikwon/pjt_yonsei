"""
Output Results Module

Enhanced RAG 기반 Multi-turn 통계 분석 시스템의 결과 저장 및 관리 모듈

이 모듈은 다음과 같은 출력 결과를 관리합니다:
- 통계 분석 결과 (JSON/CSV)
- 생성된 보고서 (Markdown/HTML/PDF)
- 시각화 결과 (PNG/SVG)
- 세션 로그 및 메타데이터
- 코드 실행 결과
"""

from .result_manager import ResultManager
from .output_formatter import OutputFormatter

__all__ = ['ResultManager', 'OutputFormatter'] 