# 파일명: core/app_context.py
"""
AppContext - 워크플로우 전체에서 공유되는 데이터 컨텍스트.
@dataclass로 전환하여 타입 안정성과 명확성을 확보합니다.
"""
from typing import Dict, Any, Optional, List
import pandas as pd
from dataclasses import dataclass, field

@dataclass
class AppContext:
    """
    워크플로우의 여러 단계에 걸쳐 데이터를 전달하기 위한 중앙 저장소.
    명시적으로 정의된 필드를 통해 데이터의 흐름을 관리합니다.
    """
    # [필수 초기 데이터]
    file_path: str
    user_request: str
    
    # [단계별 추가 데이터] - Optional과 default=None으로 선언하여 점진적으로 채워나가도록 함
    # -- Step 1: DataSelectionStep에서 추가 --
    dataframe: Optional[pd.DataFrame] = None
    dataframe_info: Optional[str] = None
    dataframe_head: Optional[str] = None
    
    # -- Step 2: AutonomousAnalysisStep에서 추가 --
    structured_request: Optional[Dict[str, Any]] = None
    retrieved_knowledge_text: Optional[str] = None
    retrieved_knowledge_raw: Optional[List[Dict[str, Any]]] = field(default_factory=list)
    analysis_plan: Optional[Dict[str, Any]] = None
    execution_results: Optional[List[Dict[str, Any]]] = field(default_factory=list)
    final_summary: Optional[Dict[str, Any]] = None
    
    # -- Step 3: ReportingStep에서 추가 --
    final_report_path: Optional[str] = None
    final_report_content: Optional[str] = None

    # -- 오류 발생 시 추가 --
    error: Optional[str] = None