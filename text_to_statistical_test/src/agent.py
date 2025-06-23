import os
import json
import re
from typing import List, Dict, Tuple

import openai
from dotenv import load_dotenv

from src.components.context import Context
from src.prompts import system_prompts

# .env 파일에서 환경 변수 로드
load_dotenv()

class Agent:
    """
    OpenAI API와의 모든 통신을 담당하는 LLM 에이전트 클래스입니다.
    """
    def __init__(self) -> None:
        """
        Agent를 초기화하고 OpenAI 클라이언트를 설정합니다.
        """
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")

    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """
        OpenAI Chat Completion API를 호출하는 비공개 헬퍼 메서드.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred while calling OpenAI API: {e}")
            return ""

    def _clean_code_response(self, response_text: str) -> Tuple[str, str]:
        """
        LLM 응답에서 <RATIONALE>과 <PYTHON_SCRIPT> 태그 사이의 내용을 추출합니다.
        
        Args:
            response_text (str): LLM이 생성한 원본 응답 텍스트.
            
        Returns:
            Tuple[str, str]: (추출된 Rationale, 정리된 Python 코드).
        """
        if not response_text:
            return "", ""
        
        rationale_match = re.search(r'<RATIONALE>(.*?)</RATIONALE>', response_text, re.DOTALL)
        code_match = re.search(r'<PYTHON_SCRIPT>(.*?)</PYTHON_SCRIPT>', response_text, re.DOTALL)
        
        rationale = rationale_match.group(1).strip() if rationale_match else ""
        code = code_match.group(1).strip() if code_match else ""
        
        if not code and not rationale:
             # XML 태그를 찾지 못한 경우 대비책
            print("Warning: Could not find XML tags. Response might be malformed.")
            return "", response_text # 원본 텍스트를 코드로 간주

        return rationale, code

    def generate_analysis_plan(self, context: Context) -> List[str]:
        """
        사용자 요청과 데이터 컨텍스트를 기반으로 통계 분석 계획을 생성합니다.
        """
        rag_context_str = "\n".join(context.rag_results)

        prompt = system_prompts.PLANNING_PROMPT.replace(
            '{user_request}', context.user_input.get('request', '')
        ).replace(
            '{data_summary}', context.data_summary
        ).replace(
            '{rag_context}', rag_context_str
        )
        
        messages = [{"role": "system", "content": prompt}]
        response_text = self._call_api(messages)
        
        plan_lines = response_text.strip().split('\n')
        plan = [re.sub(r'^\s*\d+\.\s*', '', line).strip() for line in plan_lines if line.strip()]
        return plan

    def _build_code_generation_prompt(self, task_specific_instructions: str, context: Context) -> str:
        """
        코드 생성을 위한 전체 프롬프트를 동적으로 구성합니다.
        """
        summary_history_str = context.get_summary_history()

        return system_prompts.CODE_GENERATION_PROMPT.replace(
            '{task_specific_instructions}', task_specific_instructions
        ).replace(
            '{data_summary}', context.data_summary
        ).replace(
            '{conversation_history}', summary_history_str
        )

    def generate_code_for_step(self, context: Context, current_step: str) -> Tuple[str, str]:
        """
        분석 계획의 특정 단계를 수행하기 위한 Python 코드를 생성합니다.
        """
        task_instructions = (
            f"**Full Analysis Plan**:\n{json.dumps(context.analysis_plan, indent=2, ensure_ascii=False)}\n\n"
            f"**Current Step to Implement**:\n{current_step}"
        )
        
        prompt = self._build_code_generation_prompt(task_instructions, context)
        
        messages = [{"role": "system", "content": prompt}]
        raw_response = self._call_api(messages)
        return self._clean_code_response(raw_response)

    def self_correct_code(self, context: Context, failed_step: str, failed_rationale: str, failed_code: str, error_message: str) -> Tuple[str, str]:
        """
        실패한 코드와 오류 메시지를 기반으로 코드를 자가 수정합니다.
        """
        task_instructions = (
            f"Your previous attempt to implement a step failed. You must analyze the error and provide a corrected script.\n\n"
            f"**The Goal (Original Step)**:\n{failed_step}\n\n"
            f"**Your Original Rationale**:\n{failed_rationale}\n\n"
            f"**The Code that Failed**:\n```python\n{failed_code}\n```\n\n"
            f"**The Error Message Received**:\n```\n{error_message}\n```\n\n"
            f"Please analyze your original reasoning, the failed code, and the error message. Then, provide a new, corrected version of the Python script in the required format."
        )
        
        prompt = self._build_code_generation_prompt(task_instructions, context)
        
        messages = [{"role": "system", "content": prompt}]
        raw_response = self._call_api(messages)
        return self._clean_code_response(raw_response)

    def generate_final_report(self, context: Context, final_data_shape: Tuple[int, int]) -> str:
        """
        전체 분석 과정을 요약하는 최종 보고서를 생성합니다.
        """
        summary_lines = []
        for item in context.plan_execution_summary:
            summary_lines.append(f"- {item['step']} ... **{item['status']}**")
        
        plan_summary_str = "\n".join(summary_lines)
        
        prompt = system_prompts.REPORTING_PROMPT.format(
            user_request=context.user_input.get('request', ''),
            plan_execution_summary=plan_summary_str,
            final_data_shape=f"{final_data_shape[0]} rows, {final_data_shape[1]} columns",
            conversation_history=json.dumps(context.conversation_history, indent=2, ensure_ascii=False)
        )
        
        messages = [{"role": "system", "content": prompt}]
        return self._call_api(messages)