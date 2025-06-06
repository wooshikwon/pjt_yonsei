# 파일명: services/llm/llm_response_parser.py

import json
import re
import logging
from typing import Dict, Any, List, Union

# [UTIL-REQ] error_handler.py의 ParsingException, ErrorCode 클래스가 필요합니다.
from utils.error_handler import ParsingException, ErrorCode

logger = logging.getLogger(__name__)

class LLMResponseParser:
    """LLM의 텍스트 응답을 파싱하여 구조화된 데이터로 변환합니다."""

    def extract_json(self, response_text: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        텍스트에서 JSON 객체 또는 배열을 추출합니다.
        마크다운 코드 블록(```json ... ```)을 우선적으로 처리합니다.
        """
        if not isinstance(response_text, str):
            raise ParsingException("파싱할 응답이 문자열이 아닙니다.", ErrorCode.VALIDATION_ERROR)
            
        logger.debug(f"JSON 추출 시도: {response_text[:200]}...")

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
            
            last_end_char = response_text.rfind(end_char)
            if last_end_char == -1:
                 raise ParsingException("응답에서 JSON 닫는 문자를 찾을 수 없습니다.", ErrorCode.LLM_RESPONSE_ERROR)

            json_str = response_text[start_index : last_end_char + 1]
            return json.loads(json_str)

        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 실패: {e}. 응답: {response_text}")
            raise ParsingException(f"LLM 응답을 JSON으로 파싱하는 데 실패했습니다: {e}", ErrorCode.LLM_RESPONSE_ERROR)
        except Exception as e:
            logger.error(f"JSON 추출 중 예기치 않은 오류 발생: {e}")
            raise ParsingException(f"JSON 추출 중 알 수 없는 오류 발생: {e}", ErrorCode.UNKNOWN_ERROR)