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
from .agent_reporting import AgentReportingPipeline

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
    'AgentReportingPipeline',
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
        'class': 'DataSelectionPipeline',
        'description': '데이터 파일 선택',
        'implemented': True
    },
    2: {
        'name': 'user_request',
        'class': 'UserRequestPipeline', 
        'description': '사용자 자연어 요청 처리',
        'implemented': True
    },
    3: {
        'name': 'data_summary',
        'class': 'DataSummaryPipeline',
        'description': '데이터 요약 및 기본 통계',
        'implemented': True
    },
    4: {
        'name': 'analysis_proposal',
        'class': 'AnalysisProposalPipeline',
        'description': '시스템 분석 제안',
        'implemented': False
    },
    5: {
        'name': 'method_selection',
        'class': 'MethodSelectionPipeline',
        'description': '사용자 분석 방식 선택',
        'implemented': False
    },
    6: {
        'name': 'agent_analysis',
        'class': 'AgentAnalysisPipeline',
        'description': 'RAG를 활용한 LLM AGENT 데이터 분석',
        'implemented': False
    },
    7: {
        'name': 'agent_testing',
        'class': 'AgentTestingPipeline',
        'description': 'LLM AGENT 통계 검정 (AGENTIC FLOW)',
        'implemented': False
    },
    8: {
        'name': 'agent_reporting',
        'class': 'AgentReportingPipeline',
        'description': 'LLM AGENT 보고서 생성',
        'implemented': False
    }
} 