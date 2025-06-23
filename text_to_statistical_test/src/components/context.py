from typing import Any, Dict, List, Optional

class Context:
    """
    LLM 에이전트의 "작업 기억 공간(Working Memory)" 역할을 하는 중앙 상태 관리자 클래스입니다.
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
        """
        분석 계획 실행 요약에 단계별 결과를 추가합니다.

        Args:
            step (str): 분석 계획의 단계 이름.
            status (str): 단계 실행 상태 (예: "SUCCESS", "FAILED").
        """
        self.plan_execution_summary.append({"step": step, "status": status})

    def add_rationale_history(self, rationale: str) -> None:
        """
        대화 기록에 LLM이 생성한 Rationale을 추가합니다.
        
        Args:
            rationale (str): LLM이 코드 생성 전에 제시한 논리적 근거.
        """
        if rationale: # Rationale이 비어있지 않은 경우에만 추가
            self.conversation_history.append({'type': 'rationale', 'content': rationale})

    def add_code_history(self, code: str) -> None:
        """
        대화 기록에 LLM이 생성한 코드를 추가합니다.
        
        Args:
            code (str): LLM이 생성한 Python 코드.
        """
        self.conversation_history.append({'type': 'code', 'content': code})

    def add_output_history(self, output: str) -> None:
        """
        대화 기록에 코드 실행 결과를 추가합니다.
        
        Args:
            output (str): 코드 실행으로 발생한 표준 출력 또는 오류 메시지.
        """
        self.conversation_history.append({'type': 'output', 'content': output})

    def get_summary_history(self) -> str:
        """
        코드 생성을 위해 대화 기록을 간결하게 요약합니다.
        각 단계의 'Rationale'과 '최종 결과'에 초점을 맞춥니다.

        Returns:
            str: "단계 -> Rationale -> 결과" 형식의 요약된 히스토리 문자열.
        """
        if not self.plan_execution_summary:
            return "No steps have been executed yet."

        summary_lines = []
        all_rationales = [item['content'] for item in self.conversation_history if item['type'] == 'rationale']
        all_outputs = [item['content'] for item in self.conversation_history if item['type'] == 'output']
        
        rationale_cursor = 0
        output_cursor = 0

        for i, summary_item in enumerate(self.plan_execution_summary):
            step_description = summary_item['step']
            status = summary_item['status']

            # --- Rationale 추출 ---
            # 각 단계는 최소 1개의 Rationale을 가지므로, 순서대로 하나를 가져옴
            current_rationale = ""
            if rationale_cursor < len(all_rationales):
                current_rationale = all_rationales[rationale_cursor]

            # 자가 교정 시도가 있었는지 여부를 판단 (성공/실패 무관)
            is_corrected_attempt = 'Corrected' in status
            
            # 자가 교정이 있었다면 Rationale이 2개 소비되므로 커서를 2칸 이동
            num_rationales_for_step = 2 if is_corrected_attempt else 1
            rationale_cursor += num_rationales_for_step

            # --- 최종 Output 추출 ---
            num_outputs_for_step = 2 if is_corrected_attempt else 1
            step_end_output_index = output_cursor + num_outputs_for_step
            
            final_output_content = "Execution result is missing."
            if step_end_output_index <= len(all_outputs):
                final_output_content = all_outputs[step_end_output_index - 1]
            
            output_cursor = step_end_output_index
            
            summary_lines.append(
                f"--- Previous Step {i+1} ---\n"
                f"Task: {step_description}\n"
                f"Rationale: {current_rationale}\n"
                f"Final Result: {final_output_content.strip()}\n"
            )
        
        return "\n".join(summary_lines)

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