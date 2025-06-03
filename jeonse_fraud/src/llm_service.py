# src/llm_service.py

import os
import logging
import time # 재시도 간 대기시간용
from typing import Dict, Any, Optional, List, Union # Union 추가

# LLM 클라이언트 라이브러리 (예: OpenAI)
from openai import OpenAI, APIError, Timeout # 구체적인 예외 타입 임포트

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings.get('llm', {})
        self.provider = self.settings.get('provider', 'openai')
        self.model_name = self.settings.get('model_name', 'gpt-4o')
        self.temperature = float(self.settings.get('temperature', 0.2)) # float으로 명시적 변환
        self.max_tokens = int(self.settings.get('max_tokens', 3000))   # int로 명시적 변환
        self.timeout = float(self.settings.get('timeout', 180.0))      # float으로 명시적 변환

        self.api_key = os.getenv("OPENAI_API_KEY") # OpenAI 사용 가정

        if not self.api_key:
            logger.critical(f"CRITICAL: OPENAI_API_KEY environment variable not found. LLM service will not function.")
            self.client = None
        else:
            if self.provider == "openai":
                try:
                    self.client = OpenAI(api_key=self.api_key, timeout=self.timeout)
                    logger.info(f"OpenAI client initialized for model: {self.model_name}")
                except Exception as e:
                    logger.critical(f"Failed to initialize OpenAI client: {e}", exc_info=True)
                    self.client = None
            # TODO: 다른 LLM 제공업체(Google Gemini 등)를 위한 클라이언트 초기화 로직 추가
            else:
                logger.critical(f"Unsupported LLM provider specified in settings: {self.provider}")
                self.client = None
        
    def _prepare_messages(self, prompt: str, system_message: Optional[str] = None) -> List[Dict[str,str]]:
        """Helper to prepare messages list for OpenAI API"""
        messages: List[Dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate_response(self, 
                          prompt: str, 
                          system_message: Optional[str] = None,
                          max_retries: int = 2, # 설정 파일에서 관리 가능
                          retry_delay_seconds: int = 5) -> Optional[str]:
        if not self.client:
            logger.error("LLM client is not initialized (likely API key issue). Cannot generate response.")
            # 사용자에게 전달될 수 있는 오류 메시지 또는 None 반환
            return "오류: LLM 서비스 연결 실패. API 키 또는 네트워크 설정을 확인하세요."

        logger.info(f"Generating LLM response using model {self.model_name}. System message: {'Yes' if system_message else 'No'}. Prompt (first 100 chars): '{prompt[:100]}...'")
        
        messages = self._prepare_messages(prompt, system_message)

        for attempt in range(max_retries + 1):
            try:
                if self.provider == "openai":
                    logger.debug(f"Attempting OpenAI API call (attempt {attempt + 1}) with messages: {messages}")
                    chat_completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages, # type: ignore # OpenAI v1.x.x 호환성 주석 (필요시)
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    content = chat_completion.choices[0].message.content
                    
                    if content:
                        logger.info(f"LLM response received (length: {len(content)}).")
                        logger.debug(f"LLM full response: {content[:500]}{'...' if len(content) > 500 else ''}")
                        return content.strip()
                    else:
                        logger.warning("LLM returned an empty content.")
                        # 빈 응답도 재시도 대상에 포함할 수 있으나, 여기서는 None 반환
                        return None # 또는 "LLM이 빈 응답을 반환했습니다." 와 같은 메시지
                else:
                    logger.error(f"API call logic for provider '{self.provider}' not implemented.")
                    return f"오류: 설정된 LLM 제공자({self.provider})에 대한 API 호출 로직이 구현되지 않았습니다."

            except APIError as e: # OpenAI 라이브러리가 제공하는 구체적인 API 에러
                logger.error(f"OpenAI API Error (attempt {attempt + 1}/{max_retries + 1}): {e.status_code} - {e.message}", exc_info=True)
                if e.status_code == 429: # Rate limit
                    logger.warning(f"Rate limit hit. Retrying in {retry_delay_seconds * (attempt + 1)} seconds...")
                    time.sleep(retry_delay_seconds * (attempt + 1)) # 점진적 대기 시간 증가
                elif e.status_code in [500, 502, 503, 504]: # Server-side errors
                    if attempt < max_retries:
                        logger.warning(f"Server-side error. Retrying in {retry_delay_seconds} seconds...")
                        time.sleep(retry_delay_seconds)
                    else:
                        logger.error("Max retries reached for server-side error.")
                        return f"오류: LLM 서비스 서버 오류 (시도 {attempt+1}/{max_retries+1})."
                else: # Other API errors (e.g., authentication, invalid request) - 재시도 불필요
                    logger.error("Non-retryable API error.")
                    return f"오류: LLM API 오류 ({e.status_code}). 요청 또는 인증 정보를 확인하세요."
            except Timeout as e:
                logger.error(f"OpenAI API Timeout (attempt {attempt + 1}/{max_retries + 1}): {e}", exc_info=True)
                if attempt < max_retries:
                    logger.warning(f"Timeout. Retrying in {retry_delay_seconds} seconds...")
                    time.sleep(retry_delay_seconds)
                else:
                    logger.error("Max retries reached for timeout error.")
                    return "오류: LLM 서비스 응답 시간 초과."
            except Exception as e: # 기타 예외 (네트워크 문제 등)
                logger.error(f"Unexpected error calling LLM (attempt {attempt + 1}/{max_retries + 1}): {e}", exc_info=True)
                if attempt < max_retries:
                    logger.warning(f"Retrying in {retry_delay_seconds} seconds...")
                    time.sleep(retry_delay_seconds)
                else:
                    logger.error("Max retries reached for unexpected error.")
                    return "오류: LLM 서비스 호출 중 예상치 못한 오류가 발생했습니다."
        
        return None # 모든 재시도 실패 시