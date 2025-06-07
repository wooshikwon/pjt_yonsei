import logging
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path

from .llm_client import get_llm_client
from .llm_response_parser import LLMResponseParser
from config.settings import get_settings
from services.rag.rag_service import RAGService
from utils import LLMException, PromptException
from utils.json_utils import CustomJSONEncoder

logger = logging.getLogger(__name__)

class LLMService:
    """
    LLM 관련 기능들을 통합하여 고수준의 서비스를 제공하는 Facade 클래스.
    모든 메소드는 비동기로 실행됩니다.
    """
    def __init__(self, rag_service: RAGService):
        self.client = get_llm_client()
        self.parser = LLMResponseParser()
        self.rag_service = rag_service
        # 프롬프트 템플릿이 있는 기본 디렉토리 설정
        self.prompts_dir = Path(__file__).parent.parent.parent / "resources" / "prompts"
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"LLMService가 초기화되었습니다. 프롬프트 디렉토리: {self.prompts_dir}")

    def _load_prompt(self, prompt_path: str, **kwargs) -> str:
        """
        파일에서 프롬프트 템플릿을 로드하고 주어진 context로 포맷팅합니다.
        """
        try:
            full_path = self.prompts_dir / prompt_path
            template = full_path.read_text(encoding="utf-8")
            if kwargs:
                return template.format(**kwargs)
            return template
        except FileNotFoundError:
            raise PromptException(f"프롬프트 파일을 찾을 수 없습니다: {full_path}")
        except KeyError as e:
            raise PromptException(f"프롬프트 '{prompt_path}'에 필요한 키가 누락되었습니다: {e}")
        except Exception as e:
            raise PromptException(f"프롬프트 로딩 중 오류 발생: {e}")

    async def _get_structured_response(self, prompt: str, response_format: Dict[str, Any]) -> Dict[str, Any]:
        """LLM 호출 및 구조화된 응답 파싱을 위한 내부 헬퍼 함수"""
        try:
            response_text = await self.client.generate_completion(
                prompt=prompt,
                temperature=get_settings().llm.temperature,
                max_tokens=get_settings().llm.max_tokens,
                is_json=True
            )
            return self.parser.parse_json_response(response_text, response_format)
        except Exception as e:
            self.logger.error(f"LLM 응답 파싱 중 오류 발생: {e}", exc_info=True)
            # 구조화된 응답을 기대하는 곳에서 빈 딕셔너리나 적절한 기본값을 반환
            return {"error": "Failed to get structured response from LLM", "details": str(e)}

    async def structure_user_request(
        self,
        user_request: str,
        dataframe: pd.DataFrame
    ) -> Dict[str, Any]:
        """사용자 요청을 구조화된 분석 목표로 변환합니다."""
        self.logger.debug("Structuring user request using LLM.")
        system_prompt = self._load_prompt('request_structuring/system.prompt')
        human_prompt = self._load_prompt(
            'request_structuring/human.prompt',
            user_request=user_request,
            dataframe_head=dataframe.head().to_markdown(index=False),
            dataframe_info=dataframe.info.__repr__()
        )
        
        prompt = f"{system_prompt}\n\n{human_prompt}"
        
        response_model = self.parser.get_response_model("StructuredRequest")
        return await self._get_structured_response(prompt, response_format=response_model)

    async def create_analysis_plan(
        self,
        structured_request: Dict[str, Any],
        dataframe: pd.DataFrame,
        tool_definitions: List[Dict[str, Any]],
        knowledge_context: str = ""
    ) -> Dict[str, Any]:
        """LLM을 사용하여 상세한 자율 분석 계획을 생성합니다."""
        self.logger.debug("Creating analysis plan using LLM with RAG context.")

        human_prompt = self._load_prompt(
            'analysis_planner/human.prompt',
            structured_request=json.dumps(structured_request, indent=2, ensure_ascii=False, cls=CustomJSONEncoder),
            dataframe_head=dataframe.head().to_markdown(index=False)
        )

        # 시스템 프롬프트 로딩과 포맷팅을 한번에 처리
        system_prompt = self._load_prompt(
            'analysis_planner/system.prompt',
            tool_definitions=json.dumps(tool_definitions, indent=2, ensure_ascii=False, cls=CustomJSONEncoder),
            knowledge_context=knowledge_context
        )
        
        prompt = f"{system_prompt}\n\n{human_prompt}"
        
        response_model = self.parser.get_response_model("AnalysisPlan")
        return await self._get_structured_response(prompt, response_format=response_model)

    async def summarize_analysis_results(
        self,
        structured_request: Dict[str, Any],
        analysis_plan: Dict[str, Any],
        execution_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """실행된 모든 결과를 종합하여 최종 해석을 생성합니다."""
        self.logger.debug("Summarizing analysis results using LLM.")
        system_prompt = self._load_prompt('result_summarizer/system.prompt')
        human_prompt = self._load_prompt(
            'result_summarizer/human.prompt',
            structured_request=json.dumps(structured_request, indent=2, ensure_ascii=False, cls=CustomJSONEncoder),
            analysis_plan=json.dumps(analysis_plan, indent=2, ensure_ascii=False, cls=CustomJSONEncoder),
            execution_results=json.dumps(execution_results, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
        )
        
        prompt = f"{system_prompt}\n\n{human_prompt}"
        
        response_model = self.parser.get_response_model("FinalSummary")
        return await self._get_structured_response(prompt, response_format=response_model)

    async def generate_report_content(
        self,
        user_request: str,
        analysis_results: Dict[str, Any],
        visualization_paths: List[str],
        knowledge_context: str = ""
    ) -> str:
        """모든 분석 결과를 종합하여 최종 보고서의 Markdown 콘텐츠를 생성합니다."""
        self.logger.debug("Generating final report content using LLM with RAG context.")
        system_prompt = self._load_prompt('report_generator/system.prompt')
        human_prompt = self._load_prompt(
            'report_generator/human.prompt',
            user_request=user_request,
            analysis_summary=json.dumps(analysis_results, indent=2, ensure_ascii=False, cls=CustomJSONEncoder),
            visualization_paths=json.dumps(visualization_paths, indent=2, ensure_ascii=False, cls=CustomJSONEncoder),
            knowledge_context=knowledge_context
        )
        
        prompt = f"{system_prompt}\n\n{human_prompt}"
        
        # 보고서 내용은 자유 형식이므로 일반 텍스트로 받음
        try:
            return await self.client.generate_completion(
                prompt=prompt,
                temperature=get_settings().llm.temperature,
                max_tokens=get_settings().llm.max_tokens,
                is_json=False
            )
        except Exception as e:
            self.logger.error(f"보고서 콘텐츠 생성 중 오류 발생: {e}", exc_info=True)
            return f"## 보고서 생성 오류\n\n보고서 콘텐츠를 생성하는 중 오류가 발생했습니다:\n\n```\n{e}\n```"