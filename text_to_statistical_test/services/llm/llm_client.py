# 파일명: services/llm/llm_client.py

import os
import logging
from typing import Dict, Any
from openai import OpenAI, APIError, AuthenticationError, APITimeoutError, AsyncOpenAI
from functools import lru_cache
import threading

from config.settings import get_settings
from utils import LLMException
from utils.error_handler import ErrorCode

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_llm_client() -> "LLMClient":
    """LLMClient의 싱글턴 인스턴스를 반환합니다."""
    return LLMClient()

class LLMClient:
    """
    OpenAI API와의 통신을 관리하는 클라이언트 클래스.
    API 키 관리, 동기/비동기 클라이언트 제공, 기본 모델 설정 등을 담당합니다.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 중복 초기화를 방지합니다.
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        settings = get_settings()
        self.api_key = settings.llm.openai_api_key
        if not self.api_key:
            raise LLMException("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        
        self.default_model = settings.llm.default_model
        self._sync_client = OpenAI(api_key=self.api_key)
        self._async_client = AsyncOpenAI(api_key=self.api_key)
        self._initialized = True
        logger.info("LLMClient initialized successfully.")

    @property
    def sync_client(self) -> OpenAI:
        return self._sync_client

    @property
    def async_client(self) -> AsyncOpenAI:
        return self._async_client

    async def generate_completion(self,
                            prompt: str,
                            system_prompt: str = "당신은 유능한 AI 어시스턴트입니다.",
                            model: str = "gpt-4o",
                            max_tokens: int = 2048,
                            temperature: float = 0.3,
                            is_json: bool = False) -> str:
        """
        주어진 프롬프트를 사용하여 LLM으로부터 텍스트 응답을 생성합니다. (비동기)

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
            
            response = await self._async_client.chat.completions.create(
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