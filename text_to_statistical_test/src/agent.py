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

        # 만약 전체가 JSON 블록이면, code 키만 추출
        try:
            json_match = re.search(r'\{.*\}', code_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                if 'code' in parsed:
                    code_text = parsed['code']
        except json.JSONDecodeError:
            pass # Not a JSON, proceed with markdown cleaning
            
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

    def _parse_json_response(self, response_text: str) -> Dict[str, str]:
        """LLM의 JSON 응답을 다층적으로 파싱하고, 예외 발생 시 기본값을 반환합니다."""
        try:
            # 1단계: 명확한 구분자(###JSON_START###...###JSON_END###)로 추출
            match = re.search(r'###JSON_START###(.*)###JSON_END###', response_text, re.DOTALL)
            if match:
                json_string = match.group(1).strip()
                parsed = json.loads(json_string)
                if isinstance(parsed, dict) and "status" in parsed and "code" in parsed:
                    parsed['code'] = self._clean_code_response(parsed['code'])
                    return parsed

            # 2단계 (폴백): 구분자가 없을 경우, 전체 텍스트에서 JSON 객체 찾기
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_string = json_match.group(0)
                parsed = json.loads(json_string)
                if isinstance(parsed, dict) and "status" in parsed and "code" in parsed:
                    parsed['code'] = self._clean_code_response(parsed['code'])
                    return parsed
            
        except (json.JSONDecodeError, AttributeError):
            # 3단계 (최후의 수단): 모든 파싱 실패 시, 전체를 코드로 간주
            pass
        
        return {"status": "EXECUTED", "code": self._clean_code_response(response_text)}

    def generate_code_for_step(self, context: Context, current_step: str) -> Dict[str, str]:
        """
        분석 계획의 특정 단계를 수행하기 위한 Python 코드를 생성합니다.

        Args:
            context (Context): 현재 작업 컨텍스트.
            current_step (str): 현재 실행할 분석 단계.

        Returns:
            Dict[str, str]: 'status'와 'code'를 포함하는 딕셔너리.
        """
        history_str = json.dumps(context.conversation_history, indent=2, ensure_ascii=False)

        prompt_body = system_prompts.CODE_GENERATION_PROMPT.replace(
            '{analysis_plan}', '\n'.join(context.analysis_plan)
        ).replace(
            '{current_step}', current_step
        ).replace(
            '{data_summary}', context.data_summary
        ).replace(
            '{conversation_history}', history_str
        )
        prompt = prompt_body + system_prompts.CODE_GENERATION_PROMPT_EXAMPLES
        
        messages = [{"role": "system", "content": prompt}]
        raw_response = self._call_api(messages)
        return self._parse_json_response(raw_response)

    def self_correct_code(self, context: Context, failed_step: str, failed_code: str, error_message: str) -> Dict[str, str]:
        """
        실패한 코드와 오류 메시지를 기반으로 코드를 자가 수정합니다.

        Args:
            context (Context): 현재 작업 컨텍스트.
            failed_step (str): 실패한 분석 단계.
            failed_code (str): 실패한 원본 코드.
            error_message (str): 발생한 오류 메시지.

        Returns:
            Dict[str, str]: 'status'와 'code'를 포함하는 딕셔너리.
        """
        prompt = system_prompts.SELF_CORRECTION_PROMPT.replace(
            '{failed_step}', failed_step
        ).replace(
            '{failed_code}', failed_code
        ).replace(
            '{error_message}', error_message
        ).replace(
            '{data_schema}', context.data_summary
        )
        
        messages = [{"role": "system", "content": prompt}]
        raw_response = self._call_api(messages)
        return self._parse_json_response(raw_response)

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
            status_icon = "✅" if "Success" in item["status"] else "❌"
            summary_lines.append(f"- {item['step']} ... **{item['status']}** {status_icon}")
        
        plan_summary_str = "\n".join(summary_lines)
        
        prompt = system_prompts.REPORTING_PROMPT.format(
            user_request=context.user_input.get('request', ''),
            plan_execution_summary=plan_summary_str,
            final_data_shape=f"{final_data_shape[0]} rows, {final_data_shape[1]} columns",
            conversation_history=json.dumps(context.conversation_history, indent=2, ensure_ascii=False)
        )
        
        messages = [{"role": "system", "content": prompt}]
        return self._call_api(messages) 