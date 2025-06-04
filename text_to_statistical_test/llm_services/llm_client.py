"""
LLMClient: LLM API 게이트웨이

다양한 LLM 제공자(OpenAI, Anthropic 등)와의 통신을 처리하고
통계 검정 분석에 필요한 텍스트 생성 기능을 제공합니다.
"""

import logging
import time
import requests
from typing import List, Dict, Optional, Any
import os
from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """LLM 클라이언트 기본 클래스"""
    
    @abstractmethod
    def generate_text(self, prompt: str, system_prompt: str = None, 
                     temperature: float = None, stop_sequences: List[str] = None) -> str:
        pass
    
    @abstractmethod
    def generate_chat_completion(self, messages: List[Dict], 
                               temperature: float = None, stop_sequences: List[str] = None) -> str:
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API 클라이언트"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o", 
                 default_temperature: float = 0.5, max_retries: int = 3):
        self.api_key = api_key
        self.model_name = model_name
        self.default_temperature = default_temperature
        self.max_retries = max_retries
        self.base_url = "https://api.openai.com/v1"
        self.logger = logging.getLogger(__name__)
        
        # 세션 설정
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    def generate_text(self, prompt: str, system_prompt: str = None, 
                     temperature: float = None, stop_sequences: List[str] = None) -> str:
        """텍스트 생성"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        return self.generate_chat_completion(messages, temperature, stop_sequences)
    
    def generate_chat_completion(self, messages: List[Dict], 
                               temperature: float = None, stop_sequences: List[str] = None) -> str:
        """채팅 형식 API 호출"""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature or self.default_temperature,
            "max_tokens": 4000
        }
        
        if stop_sequences:
            payload["stop"] = stop_sequences
        
        for attempt in range(self.max_retries):
            try:
                response = self._session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    self._handle_api_error(response)
                    
            except requests.RequestException as e:
                self.logger.warning(f"API 요청 실패 (시도 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # 지수적 백오프
                else:
                    raise
        
        raise Exception("API 호출 최대 재시도 횟수 초과")
    
    def _handle_api_error(self, response: requests.Response):
        """API 에러 처리"""
        error_msg = f"OpenAI API 오류 {response.status_code}: {response.text}"
        self.logger.error(error_msg)
        
        if response.status_code == 429:
            raise Exception("API 요청 한도 초과. 잠시 후 다시 시도해주세요.")
        elif response.status_code == 401:
            raise Exception("API 키가 유효하지 않습니다.")
        elif response.status_code >= 500:
            raise Exception("OpenAI 서버 오류. 잠시 후 다시 시도해주세요.")
        else:
            raise Exception(error_msg)


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API 클라이언트"""
    
    def __init__(self, api_key: str, model_name: str = "claude-3-sonnet-20240229", 
                 default_temperature: float = 0.5, max_retries: int = 3):
        self.api_key = api_key
        self.model_name = model_name
        self.default_temperature = default_temperature
        self.max_retries = max_retries
        self.base_url = "https://api.anthropic.com/v1"
        self.logger = logging.getLogger(__name__)
        
        # 세션 설정
        self._session = requests.Session()
        self._session.headers.update({
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        })
    
    def generate_text(self, prompt: str, system_prompt: str = None, 
                     temperature: float = None, stop_sequences: List[str] = None) -> str:
        """텍스트 생성"""
        messages = [{"role": "user", "content": prompt}]
        return self.generate_chat_completion(messages, temperature, stop_sequences, system_prompt)
    
    def generate_chat_completion(self, messages: List[Dict], 
                               temperature: float = None, stop_sequences: List[str] = None,
                               system_prompt: str = None) -> str:
        """채팅 형식 API 호출"""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature or self.default_temperature,
            "max_tokens": 4000
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        if stop_sequences:
            payload["stop_sequences"] = stop_sequences
        
        for attempt in range(self.max_retries):
            try:
                response = self._session.post(
                    f"{self.base_url}/messages",
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["content"][0]["text"]
                else:
                    self._handle_api_error(response)
                    
            except requests.RequestException as e:
                self.logger.warning(f"API 요청 실패 (시도 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        
        raise Exception("API 호출 최대 재시도 횟수 초과")
    
    def _handle_api_error(self, response: requests.Response):
        """API 에러 처리"""
        error_msg = f"Anthropic API 오류 {response.status_code}: {response.text}"
        self.logger.error(error_msg)
        
        if response.status_code == 429:
            raise Exception("API 요청 한도 초과. 잠시 후 다시 시도해주세요.")
        elif response.status_code == 401:
            raise Exception("API 키가 유효하지 않습니다.")
        elif response.status_code >= 500:
            raise Exception("Anthropic 서버 오류. 잠시 후 다시 시도해주세요.")
        else:
            raise Exception(error_msg)


class LLMClient:
    """통합 LLM 클라이언트"""
    
    def __init__(self, provider_name: str, model_name: str = None, 
                 default_temperature: float = 0.5, max_retries: int = 3):
        """
        LLM 클라이언트 초기화
        
        Args:
            provider_name: LLM 제공자 ("openai", "anthropic")
            model_name: 모델명
            default_temperature: 기본 온도값
            max_retries: 최대 재시도 횟수
        """
        self.provider_name = provider_name.lower()
        self.logger = logging.getLogger(__name__)
        
        # 제공자별 클라이언트 초기화
        if self.provider_name == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
            
            model_name = model_name or "gpt-4o"
            self._client = OpenAIClient(api_key, model_name, default_temperature, max_retries)
            
        elif self.provider_name == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
            
            model_name = model_name or "claude-3-sonnet-20240229"
            self._client = AnthropicClient(api_key, model_name, default_temperature, max_retries)
            
        else:
            raise ValueError(f"지원하지 않는 LLM 제공자: {provider_name}")
        
        self.logger.info(f"LLM 클라이언트 초기화 완료: {provider_name}")
    
    def generate_text(self, prompt: str, system_prompt: str = None, 
                     temperature: float = None, stop_sequences: List[str] = None) -> str:
        """텍스트 생성"""
        return self._client.generate_text(prompt, system_prompt, temperature, stop_sequences)
    
    def generate_chat_completion(self, messages: List[Dict], 
                               temperature: float = None, stop_sequences: List[str] = None) -> str:
        """채팅 형식 API 호출"""
        return self._client.generate_chat_completion(messages, temperature, stop_sequences)
    
    def is_available(self) -> bool:
        """API 사용 가능 여부 확인"""
        try:
            test_response = self.generate_text("테스트", temperature=0.1)
            return bool(test_response)
        except Exception as e:
            self.logger.error(f"LLM API 연결 테스트 실패: {e}")
            return False 