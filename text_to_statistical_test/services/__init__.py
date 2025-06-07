"""
서비스(Engine) 계층의 진입점.
각 서비스의 Facade 클래스와 주요 인스턴스를 제공합니다.

Services Package

외부 서비스 연동 및 핵심 비즈니스 로직
- llm: LLM 관련 서비스
- statistics: 통계 분석 로직
- code_executor: 안전한 코드 실행 환경
"""

from .llm.llm_service import LLMService
from .rag.rag_service import RAGService
from .statistics.stats_service import StatisticsService
from .reporting.report_service import ReportService
from .code_executor.safe_code_runner import SafeCodeRunner

# Initialize services
rag_service = RAGService(knowledge_base_dir="knowledge_base")
llm_service = LLMService(rag_service=rag_service)
statistics_service = StatisticsService()
report_service = ReportService()

# Expose service instances
__all__ = [
    "llm_service",
    "rag_service",
    "statistics_service",
    "report_service",
    "SafeCodeRunner",
]