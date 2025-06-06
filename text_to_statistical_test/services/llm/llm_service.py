import logging
from typing import Dict, Any, List

from .llm_client import LLMClient
from .prompt_engine import PromptEngine
from .llm_response_parser import LLMResponseParser
# [UTIL-REQ] error_handler.py의 handle_error 함수가 필요합니다.
from utils.error_handler import handle_error

logger = logging.getLogger(__name__)

class LLMService:
    """
    LLM 관련 기능들을 통합하여 고수준의 서비스를 제공하는 Facade 클래스.
    """
    def __init__(self):
        self.client = LLMClient()
        self.prompter = PromptEngine()
        self.parser = LLMResponseParser()
        logger.info("LLM 서비스가 초기화되었습니다.")

    def interpret_user_request(self, request: str, data_context: str) -> Dict[str, Any]:
        """(2단계용) 사용자 요청을 해석하여 구조화된 목표를 반환합니다."""
        try:
            prompt = self.prompter.create_prompt(
                'interpret_user_request',
                {'user_request': request, 'data_context': data_context}
            )
            response_text = self.client.generate_completion(prompt, is_json=True)
            return self.parser.extract_json(response_text)
        except Exception as e:
            return handle_error(e, default_return={"error": str(e)})

    def create_detailed_analysis_plan(self, structured_request: Dict) -> Dict[str, Any]:
        """(3단계용) 사용자 요청을 바탕으로 상세한 자율 분석 계획을 수립합니다."""
        try:
            prompt = self.prompter.create_prompt(
                'create_detailed_analysis_plan',
                {'structured_request': str(structured_request)}
            )
            response_text = self.client.generate_completion(prompt, temperature=0.1, is_json=True)
            plan = self.parser.extract_json(response_text)
            plan['request'] = structured_request # 원본 요청 정보를 계획에 추가
            return plan
        except Exception as e:
            return handle_error(e, default_return={"error": str(e)})

    def recommend_visualizations(self, analysis_results: Dict) -> List[Dict[str, Any]]:
        """(4단계용) 분석 결과에 가장 적합한 시각화 리스트를 추천합니다."""
        try:
            prompt = self.prompter.create_prompt(
                'recommend_visualizations',
                {'analysis_results': str(analysis_results)}
            )
            response_text = self.client.generate_completion(prompt, temperature=0.2, is_json=True)
            return self.parser.extract_json(response_text)
        except Exception as e:
            return handle_error(e, default_return=[])

    def generate_report_narrative(self, results: Dict, plan: Dict, artifacts: List) -> Dict[str, str]:
        """(5단계용) 모든 결과를 종합하여 최종 보고서의 서술부를 생성합니다."""
        try:
            context = {
                'final_plan': str(plan),
                'analysis_results': str(results),
                'visual_artifacts': ", ".join(artifacts)
            }
            prompt = self.prompter.create_prompt('generate_report_narrative', context)
            response_text = self.client.generate_completion(prompt, temperature=0.5, max_tokens=3000, is_json=True)
            return self.parser.extract_json(response_text)
        except Exception as e:
            return handle_error(e, default_return={"error": str(e)})