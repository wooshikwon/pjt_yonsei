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
    시스템의 "두뇌" 역할을 하며, 프롬프트 포매팅, API 호출, 응답 파싱을 수행합니다.
    """
    def __init__(self) -> None:
        """
        Agent를 초기화하고 OpenAI 클라이언트를 설정합니다.
        API 키는 환경 변수 'OPENAI_API_KEY'에서 자동으로 로드됩니다.
        """
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """
        OpenAI Chat Completion API를 호출하는 비공개 헬퍼 메서드.

        Args:
            messages (List[Dict[str, str]]): API에 전달할 메시지 목록.

        Returns:
            str: API 응답의 내용.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred while calling OpenAI API: {e}")
            return ""

    def _clean_code_response(self, code_text: str) -> str:
        """
        LLM 응답에서 markdown 백틱과 불필요한 텍스트를 제거하여 순수한 Python 코드만 추출합니다.
        
        Args:
            code_text (str): LLM이 생성한 원본 응답 텍스트.
            
        Returns:
            str: 정리된 Python 코드.
        """
        if not code_text:
            return ""

        # 'Python Code:' 블록 이후의 내용만 추출
        # rfind를 사용하여 예시에 있는 헤더가 아닌, 실제 응답의 마지막 헤더를 찾음
        code_block_header = "**Python Code:**"
        header_index = code_text.rfind(code_block_header)
        
        if header_index != -1:
            code_text = code_text[header_index + len(code_block_header):]
            
        # markdown 코드 블록 제거 (```python ... ```)
        lines = code_text.strip().split('\n')
        cleaned_lines = []
        in_code_block = False

        # 코드 블록 시작/끝이 있는지 먼저 확인
        has_backticks = any('```' in line for line in lines)

        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            
            # 백틱이 있는 경우, 블록 내부의 코드만 포함
            # 백틱이 없는 경우, 모든 라인을 코드로 간주
            if not has_backticks or in_code_block:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()

    def generate_analysis_plan(self, context: Context) -> List[str]:
        """
        사용자 요청과 데이터 컨텍스트를 기반으로 통계 분석 계획을 생성합니다.

        Args:
            context (Context): 현재 작업 컨텍스트.

        Returns:
            List[str]: 단계별 분석 계획 목록.
        """
        prompt = system_prompts.PLANNING_PROMPT.replace(
            '{user_request}', context.user_input.get('request', '')
        ).replace(
            '{data_summary}', context.data_summary
        ).replace(
            '{rag_context}', '\n'.join(context.rag_results)
        )
        
        messages = [{"role": "system", "content": prompt}]
        response_text = self._call_api(messages)
        
        # 응답을 줄바꿈 기준으로 분리하고, 각 줄의 앞머리에 있는 숫자 목록(e.g., "1. ") 제거
        plan_lines = response_text.strip().split('\n')
        plan = [re.sub(r'^\s*\d+\.\s*', '', line).strip() for line in plan_lines if line.strip()]
        return plan

    def _build_code_generation_prompt(self, task_specific_instructions: str, context: Context) -> str:
        """
        코드 생성을 위한 전체 프롬프트를 동적으로 구성합니다.
        
        Args:
            task_specific_instructions (str): 작업별 지침 (코드 생성 또는 수정).
            context (Context): 현재 작업 컨텍스트.
            
        Returns:
            str: 완성된 전체 프롬프트 문자열.
        """
        history_str = json.dumps(context.conversation_history, indent=2, ensure_ascii=False)

        return system_prompts.CODE_GENERATION_PROMPT.replace(
            '{task_specific_instructions}', task_specific_instructions
        ).replace(
            '{data_summary}', context.data_summary
        ).replace(
            '{conversation_history}', history_str
        )

    def generate_code_for_step(self, context: Context, current_step: str) -> str:
        """
        분석 계획의 특정 단계를 수행하기 위한 Python 코드를 생성합니다.

        Args:
            context (Context): 현재 작업 컨텍스트.
            current_step (str): 현재 실행할 분석 단계.

        Returns:
            str: LLM이 생성한 순수 Python 코드.
        """
        task_instructions = (
            f"**Full Analysis Plan**:\n{json.dumps(context.analysis_plan, indent=2)}\n\n"
            f"**Current Step to Implement**:\n{current_step}"
        )
        
        prompt = self._build_code_generation_prompt(task_instructions, context)
        
        messages = [{"role": "system", "content": prompt}]
        raw_response = self._call_api(messages)
        return self._clean_code_response(raw_response)

    def self_correct_code(self, context: Context, failed_step: str, failed_code: str, error_message: str) -> str:
        """
        실패한 코드와 오류 메시지를 기반으로 코드를 자가 수정합니다.

        Args:
            context (Context): 현재 작업 컨텍스트.
            failed_step (str): 실패한 분석 단계.
            failed_code (str): 실패한 원본 코드.
            error_message (str): 발생한 오류 메시지.

        Returns:
            str: LLM이 생성한 순수 Python 코드.
        """
        task_instructions = (
            f"Your previous attempt failed. Here is the context:\n\n"
            f"**The Goal (Original Step)**:\n{failed_step}\n\n"
            f"**The Failed Code**:\n```python\n{failed_code}\n```\n\n"
            f"**The Error Message**:\n```\n{error_message}\n```\n\n"
            f"Please provide a corrected version of the Python script."
        )
        
        prompt = self._build_code_generation_prompt(task_instructions, context)
        
        messages = [{"role": "system", "content": prompt}]
        raw_response = self._call_api(messages)
        return self._clean_code_response(raw_response)

    def generate_final_report(self, context: Context, final_data_shape: Tuple[int, int]) -> str:
        """
        전체 분석 과정을 요약하는 최종 보고서를 생성합니다.

        Args:
            context (Context): 최종 작업 컨텍스트.
            final_data_shape (Tuple[int, int]): 전처리 완료 후 데이터의 최종 형태 (행, 열).

        Returns:
            str: Markdown 형식의 최종 보고서.
        """
        # plan_execution_summary를 Markdown 형식의 문자열로 변환
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