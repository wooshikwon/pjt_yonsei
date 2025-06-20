from typing import Any, Dict, List, Optional

class Context:
    """
    LLM 에이전트의 "작업 기억 공간(Working Memory)" 역할을 하는 중앙 상태 관리자 클래스입니다.
    시스템의 모든 상태와 데이터를 구조화하여 저장하고, 각 컴포넌트가 일관된 상태를 공유하도록 합니다.
    """
    def __init__(self) -> None:
        """Context 객체를 초기화합니다."""
        self.user_input: Dict[str, str] = {}
        self.rag_results: List[str] = []
        self.data_summary: str = ""
        self.analysis_plan: List[str] = []
        self.plan_execution_summary: List[Dict[str, str]] = []
        self.conversation_history: List[Dict[str, str]] = []
        self.final_report: Optional[str] = None

    def set_user_input(self, file_path: str, request: str) -> None:
        """
        사용자의 초기 입력을 설정합니다.

        Args:
            file_path (str): 사용자가 제공한 데이터 파일의 경로.
            request (str): 사용자의 자연어 분석 요청.
        """
        self.user_input = {'file_path': file_path, 'request': request}

    def add_rag_result(self, result: str) -> None:
        """
        RAG 검색 결과를 추가합니다.

        Args:
            result (str): 지식 베이스에서 검색된 관련 정보.
        """
        self.rag_results.append(result)

    def set_data_summary(self, summary: str) -> None:
        """
        데이터 프로파일러가 생성한 요약 정보를 설정합니다.

        Args:
            summary (str): 데이터 요약 정보가 담긴 Markdown 문자열.
        """
        self.data_summary = summary

    def set_analysis_plan(self, plan: List[str]) -> None:
        """
        에이전트가 생성한 통계 분석 계획을 설정합니다.

        Args:
            plan (List[str]): 단계별 분석 계획 리스트.
        """
        self.analysis_plan = plan

    def add_step_to_summary(self, step: str, status: str) -> None:
        """분석 계획 실행 요약에 단계별 결과를 추가합니다."""
        self.plan_execution_summary.append({"step": step, "status": status})

    def add_to_history(self, entry: Dict[str, str]) -> None:
        """
        대화 기록에 새 항목을 추가합니다.

        Args:
            entry (Dict[str, str]): 'role'과 'content' 또는 'code' 등을 포함하는 대화 조각.
        """
        self.conversation_history.append(entry)

    def set_final_report(self, report: str) -> None:
        """
        최종 분석 보고서를 설정합니다.

        Args:
            report (str): Markdown 형식의 최종 보고서.
        """
        self.final_report = report

    def get_full_context(self) -> Dict[str, Any]:
        """
        현재까지 축적된 모든 컨텍스트 정보를 반환합니다.

        Returns:
            Dict[str, Any]: 컨텍스트 객체의 모든 속성을 담은 딕셔너리.
        """
        return {
            "user_input": self.user_input,
            "rag_results": self.rag_results,
            "data_summary": self.data_summary,
            "analysis_plan": self.analysis_plan,
            "plan_execution_summary": self.plan_execution_summary,
            "conversation_history": self.conversation_history,
            "final_report": self.final_report,
        } 