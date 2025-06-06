"""
Pipeline Module

8단계 파이프라인 시스템을 위한 모듈 패키지
각 단계별로 독립적인 파이프라인 클래스 제공
"""

from .data_selection import DataSelectionStep
from .user_request import UserRequestStep
from .data_summary import DataSummaryStep
from .analysis_proposal import AnalysisProposalStep
from .user_selection import UserSelectionStep
from .agent_analysis import AgentAnalysisStep
from .agent_execution import AgentExecutionStep
from .agent_reporting import AgentReportingStep

# 추후 추가될 파이프라인들
# from .analysis_proposal import AnalysisProposalPipeline
# from .method_selection import MethodSelectionPipeline
# from .agent_analysis import AgentAnalysisPipeline
# from .agent_testing import AgentTestingPipeline
# from .agent_reporting import AgentReportingPipeline

__all__ = [
    'DataSelectionStep',
    'UserRequestStep', 
    'DataSummaryStep',
    'AnalysisProposalStep',
    'UserSelectionStep',
    'AgentAnalysisStep',
    'AgentExecutionStep',
    'AgentReportingStep',
    # 'AnalysisProposalPipeline',
    # 'MethodSelectionPipeline',
    # 'AgentAnalysisPipeline',
    # 'AgentTestingPipeline',
    # 'AgentReportingPipeline'
]

# 파이프라인 단계 정보
PIPELINE_STEPS = {
    1: {
        'name': 'data_selection',
        'class': 'DataSelectionStep',
        'description': '데이터 파일 선택',
        'implemented': True
    },
    2: {
        'name': 'user_request',
        'class': 'UserRequestStep', 
        'description': '사용자 자연어 요청 처리',
        'implemented': True
    },
    3: {
        'name': 'data_summary',
        'class': 'DataSummaryStep',
        'description': '데이터 요약 및 기본 통계',
        'implemented': True
    },
    4: {
        'name': 'analysis_proposal',
        'class': 'AnalysisProposalStep',
        'description': '시스템 분석 제안',
        'implemented': True
    },
    5: {
        'name': 'user_selection',
        'class': 'UserSelectionStep',
        'description': '사용자 분석 방식 선택',
        'implemented': True
    },
    6: {
        'name': 'agent_analysis',
        'class': 'AgentAnalysisStep',
        'description': 'RAG를 활용한 LLM AGENT 데이터 분석',
        'implemented': True
    },
    7: {
        'name': 'agent_execution',
        'class': 'AgentExecutionStep',
        'description': 'LLM AGENT 통계 검정 (AGENTIC FLOW)',
        'implemented': True
    },
    8: {
        'name': 'agent_reporting',
        'class': 'AgentReportingStep',
        'description': 'LLM AGENT 보고서 생성',
        'implemented': True
    }
} 