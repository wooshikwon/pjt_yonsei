import os
import json
from typing import List, Dict

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
            
        # markdown 코드 블록 제거
        lines = code_text.strip().split('\n')
        cleaned_lines = []
        in_code_block = False
        
        for line in lines:
            # 코드 블록 시작/끝 감지
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            
            # 코드 블록 내부이거나 코드 블록이 없는 경우 모든 라인 포함
            if in_code_block or '```' not in code_text:
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
        prompt = system_prompts.PLANNING_PROMPT.format(
            user_request=context.user_input.get('request', ''),
            data_schema=str(context.data_info.get('schema', {})),
            rag_context='\n'.join(context.rag_results)
        )
        
        messages = [{"role": "system", "content": prompt}]
        response_text = self._call_api(messages)
        
        # 응답을 줄바꿈 기준으로 분리하여 리스트로 변환
        plan = [step.strip() for step in response_text.strip().split('\n') if step.strip()]
        return plan

    def generate_code_for_step(self, context: Context, current_step: str) -> str:
        """
        분석 계획의 특정 단계를 수행하기 위한 Python 코드를 생성합니다.

        Args:
            context (Context): 현재 작업 컨텍스트.
            current_step (str): 현재 실행할 분석 단계.

        Returns:
            str: 생성된 Python 코드.
        """
        prompt = system_prompts.CODE_GENERATION_PROMPT.format(
            analysis_plan='\n'.join(context.analysis_plan),
            current_step=current_step
        )
        
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
            str: 수정된 Python 코드.
        """
        prompt = system_prompts.SELF_CORRECTION_PROMPT.format(
            failed_step=failed_step,
            failed_code=failed_code,
            error_message=error_message,
            data_schema=str(context.data_info.get('schema', {}))
        )
        
        messages = [{"role": "system", "content": prompt}]
        raw_response = self._call_api(messages)
        return self._clean_code_response(raw_response)

    def generate_final_report(self, context: Context) -> str:
        """
        전체 분석 과정을 요약하는 최종 보고서를 생성합니다.

        Args:
            context (Context): 최종 작업 컨텍스트.

        Returns:
            str: Markdown 형식의 최종 보고서.
        """
        prompt = system_prompts.REPORTING_PROMPT.format(
            user_request=context.user_input.get('request', ''),
            conversation_history=json.dumps(context.conversation_history, indent=2, ensure_ascii=False)
        )
        
        messages = [{"role": "system", "content": prompt}]
        return self._call_api(messages) 