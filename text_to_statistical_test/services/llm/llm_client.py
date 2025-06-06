# 파일명: services/llm/llm_client.py

import os
import logging
from typing import Dict, Any
from openai import OpenAI, APIError, AuthenticationError, APITimeoutError

# [UTIL-REQ] error_handler.py의 LLMException, ErrorCode 클래스가 필요합니다.
from utils.error_handler import LLMException, ErrorCode

logger = logging.getLogger(__name__)

class LLMClient:
    """
    LLM API와의 통신을 담당하는 저수준 클라이언트.
    OpenAI API를 기준으로 작성되었습니다.
    """
    def __init__(self):
        """LLM 클라이언트 초기화 및 API 키 로드"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
            
            self.client = OpenAI(api_key=api_key)
            logger.info("OpenAI 클라이언트가 성공적으로 초기화되었습니다.")
            
        except Exception as e:
            logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
            raise LLMException(
                "LLM 클라이언트 초기화에 실패했습니다. API 키 설정을 확인하세요.",
                error_code=ErrorCode.CONFIGURATION_ERROR
            ) from e

    def generate_completion(self,
                            prompt: str,
                            system_prompt: str = "당신은 유능한 AI 어시스턴트입니다.",
                            model: str = "gpt-4o",
                            max_tokens: int = 2048,
                            temperature: float = 0.3,
                            is_json: bool = False) -> str:
        """
        주어진 프롬프트를 사용하여 LLM으로부터 텍스트 응답을 생성합니다.

        Args:
            prompt: 사용자 요청이 포함된 주 프롬프트.
            system_prompt: AI의 역할과 행동을 정의하는 시스템 프롬프트.
            model: 사용할 LLM 모델.
            max_tokens: 생성할 최대 토큰 수.
            temperature: 생성 결과의 창의성 제어 (0.0 ~ 1.0).
            is_json: 응답 형식을 JSON으로 강제할지 여부.

        Returns:
            LLM이 생성한 텍스트 응답.
        """
        logger.debug(f"LLM API 호출 시작: model={model}, is_json={is_json}")
        try:
            response_format = {"type": "json_object"} if is_json else {"type": "text"}
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format
            )
            
            content = response.choices[0].message.content
            if not content:
                raise LLMException("LLM으로부터 빈 응답을 받았습니다.", ErrorCode.LLM_RESPONSE_ERROR)
            
            logger.debug(f"LLM API 응답 수신 완료 (토큰 사용량: {response.usage})")
            return content

        except AuthenticationError as e:
            logger.error(f"OpenAI API 인증 오류: {e}")
            raise LLMException("OpenAI API 인증에 실패했습니다.", ErrorCode.LLM_API_ERROR, original_exception=e)
        
        except APITimeoutError as e:
            logger.error(f"OpenAI API 타임아웃 오류: {e}")
            raise LLMException("OpenAI API 요청 시간이 초과되었습니다.", ErrorCode.LLM_TIMEOUT_ERROR, original_exception=e)
            
        except APIError as e:
            logger.error(f"OpenAI API 오류: {e.status_code} - {e.response}")
            raise LLMException(f"OpenAI API 오류 발생 (코드: {e.status_code}).", ErrorCode.LLM_API_ERROR, original_exception=e)
        
        except Exception as e:
            logger.error(f"LLM 응답 생성 중 예기치 않은 오류 발생: {e}")
            raise LLMException("LLM 응답 생성 중 알 수 없는 오류가 발생했습니다.", ErrorCode.UNKNOWN_ERROR, original_exception=e)