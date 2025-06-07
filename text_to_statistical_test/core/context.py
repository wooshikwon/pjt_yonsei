"""
AppContext - 워크플로우 전체에서 공유되는 데이터 컨텍스트.
"""
from typing import Dict, Any, Optional

class AppContext(dict):
    """
    워크플로우의 여러 단계에 걸쳐 데이터를 전달하기 위한 중앙 저장소.
    딕셔너리처럼 작동하지만, 타입 힌팅과 명확성을 위해 별도 클래스로 정의합니다.
    
    필수 초기 데이터:
    - file_path (str): 사용자가 업로드한 원본 데이터 파일 경로.
    - user_request (str): 사용자의 자연어 분석 요청.
    
    단계별 추가 데이터:
    - dataframe (pd.DataFrame): 로드되고 전처리된 데이터.
    - structured_request (Dict): LLM이 구조화한 분석 목표.
    - analysis_plan (Dict): 자율 분석 에이전트가 수립한 계획.
    - execution_results (List[Dict]): 계획 실행 결과.
    - final_summary (Dict): 최종 분석 요약.
    - final_report_path (str): 생성된 최종 보고서 파일 경로.
    """
    def __init__(self, **kwargs: Any):
        super().__init__(kwargs)

    def __getattr__(self, key: str) -> Optional[Any]:
        return self.get(key)

    def __setattr__(self, key: str, value: Any):
        self[key] = value

    def update(self, other: Dict[str, Any]):
        """다른 딕셔너리의 내용으로 컨텍스트를 업데이트합니다."""
        super().update(other) 