# 파일명: services/llm/llm_response_parser.py

import json
import re
import logging
from typing import Dict, Any, List, Union

from utils import ParsingException
from utils.error_handler import ErrorCode

logger = logging.getLogger(__name__)

class LLMResponseParser:
    """LLM의 텍스트 응답을 파싱하여 구조화된 데이터로 변환합니다."""

    def __init__(self):
        self._response_models = {
            "StructuredRequest": {
                "type": "object",
                "properties": {
                    "user_request": {"type": "string"},
                    "hypotheses": {
                        "type": "object",
                        "properties": {
                            "primary_hypothesis": {"type": "string"},
                            "null_hypothesis": {"type": "string"},
                            "alternative_hypothesis": {"type": "string"},
                        },
                        "required": ["primary_hypothesis", "null_hypothesis", "alternative_hypothesis"],
                    },
                    "variables": {
                        "type": "object",
                        "properties": {
                            "dependent_variable": {"type": "string"},
                            "independent_variables": {"type": "array", "items": {"type": "string"}},
                            "categorical_variables": {"type": "array", "items": {"type": "string"}},
                            "numerical_variables": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["dependent_variable", "independent_variables", "categorical_variables", "numerical_variables"],
                    },
                    "analysis_type": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string"},
                            "recommended_test": {"type": "string"},
                        },
                        "required": ["category", "recommended_test"],
                    },
                    "summary_for_agent": {"type": "string"},
                },
                "required": ["user_request", "hypotheses", "variables", "analysis_type", "summary_for_agent"],
            },
            # AnalysisPlan 및 다른 모델 스키마도 필요 시 여기에 추가할 수 있습니다.
            "AnalysisPlan": {"type": "object"}, 
            "FinalSummary": {"type": "object"},
        }

    def get_response_model(self, model_name: str) -> Dict[str, Any]:
        """정의된 응답 모델의 JSON 스키마를 반환합니다."""
        model = self._response_models.get(model_name)
        if not model:
            raise ParsingException(f"'{model_name}'에 대한 응답 모델이 정의되지 않았습니다.", ErrorCode.MODEL_NOT_DEFINED)
        return model

    def parse_json_response(self, response_text: str, response_format: Dict[str, Any]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        텍스트에서 JSON 객체 또는 배열을 추출합니다.
        마크다운 코드 블록(```json ... ```)을 우선적으로 처리합니다.
        (기존 extract_json 메서드의 이름을 변경)
        """
        # TODO: response_format 스키마를 사용한 응답 검증(validation) 로직 추가
        
        if not isinstance(response_text, str):
            raise ParsingException("파싱할 응답이 문자열이 아닙니다.", ErrorCode.VALIDATION_ERROR)
            
        logger.debug(f"JSON 추출 시도:\n{response_text[:500]}...")

        try:
            # 패턴 1: ```json ... ``` 코드 블록 찾기
            match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text)
            if match:
                json_str = match.group(1).strip()
                return json.loads(json_str)

            # 패턴 2: 일반적인 JSON 객체 또는 배열 찾기 (가장 바깥쪽 {} 또는 []를 찾음)
            first_brace = response_text.find('{')
            first_bracket = response_text.find('[')
            
            if first_brace == -1 and first_bracket == -1:
                return json.loads(response_text) # JSON 모드로 응답 시 코드 블록이 없을 수 있음
            
            start_index = -1
            if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
                start_index = first_brace
                start_char, end_char = '{', '}'
            elif first_bracket != -1:
                start_index = first_bracket
                start_char, end_char = '[', ']'
            
            if start_index == -1:
                raise ParsingException("응답에서 JSON 시작 문자를 찾을 수 없습니다.", ErrorCode.PARSING_ERROR)

            last_end_char = response_text.rfind(end_char)
            if last_end_char == -1:
                 raise ParsingException("응답에서 JSON 닫는 문자를 찾을 수 없습니다.", ErrorCode.PARSING_ERROR)

            json_str = response_text[start_index : last_end_char + 1]
            return json.loads(json_str)

        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 실패: {e}. 응답: {response_text}")
            raise ParsingException(f"LLM 응답을 JSON으로 파싱하는 데 실패했습니다: {e}", ErrorCode.PARSING_ERROR)
        except Exception as e:
            logger.error(f"JSON 추출 중 예기치 않은 오류 발생: {e}")
            raise ParsingException(f"JSON 추출 중 알 수 없는 오류 발생: {e}", ErrorCode.UNKNOWN_ERROR)