"""
LLM 클라이언트 모듈

OpenAI API와의 통신을 처리하고
다양한 요청 유형에 대한 통합된 인터페이스를 제공합니다.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod

import openai
from openai import OpenAI

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """LLM 클라이언트의 기본 추상 클래스"""
    
    def __init__(self, api_key: str, model_name: str, 
                 default_temperature: float = 0.7, max_retries: int = 3):
        self.api_key = api_key
        self.model_name = model_name
        self.default_temperature = default_temperature
        self.max_retries = max_retries
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], 
                         temperature: Optional[float] = None,
                         max_tokens: Optional[int] = None,
                         **kwargs) -> Dict[str, Any]:
        """LLM으로부터 응답을 생성합니다"""
        pass
    
    def _handle_api_error(self, error: Exception, attempt: int) -> bool:
        """API 오류를 처리하고 재시도 여부를 결정합니다"""
        if attempt < self.max_retries:
            wait_time = 2 ** attempt  # 지수 백오프
            self.logger.warning(f"API 요청 실패 (시도 {attempt + 1}/{self.max_retries}): {error}")
            self.logger.info(f"{wait_time}초 대기 후 재시도...")
            time.sleep(wait_time)
            return True
        return False


class OpenAIClient(BaseLLMClient):
    """OpenAI API 클라이언트"""
    
    def __init__(self, api_key: str, model_name: str, 
                 default_temperature: float = 0.7, max_retries: int = 3):
        super().__init__(api_key, model_name, default_temperature, max_retries)
        self.client = OpenAI(api_key=api_key)
        self.logger.info(f"OpenAI 클라이언트 초기화 완료 - 모델: {model_name}")
    
    def generate_response(self, messages: List[Dict[str, str]], 
                         temperature: Optional[float] = None,
                         max_tokens: Optional[int] = None,
                         **kwargs) -> Dict[str, Any]:
        """OpenAI API를 사용하여 응답을 생성합니다"""
        
        if temperature is None:
            temperature = self.default_temperature
            
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                
                return {
                    "content": response.choices[0].message.content,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    },
                    "model": response.model,
                    "finish_reason": response.choices[0].finish_reason
                }
                
            except openai.RateLimitError as e:
                if not self._handle_api_error(e, attempt):
                    raise Exception("OpenAI API 요청 한도 초과. 잠시 후 다시 시도해주세요.")
            except openai.APIError as e:
                if not self._handle_api_error(e, attempt):
                    raise Exception(f"OpenAI API 오류: {str(e)}")
            except Exception as e:
                if not self._handle_api_error(e, attempt):
                    raise Exception(f"예상치 못한 오류: {str(e)}")
        
        raise Exception("최대 재시도 횟수를 초과했습니다.")


class LLMClient:
    """
    통합 LLM 클라이언트
    
    OpenAI API를 사용한 언어 모델과의 상호작용을 관리합니다.
    """
    
    def __init__(self, provider_name: str = "openai", 
                 model_name: Optional[str] = None,
                 default_temperature: float = 0.7,
                 max_retries: int = 3):
        """
        LLM 클라이언트를 초기화합니다.
        
        Args:
            provider_name: LLM 제공자 ("openai")
            model_name: 사용할 모델명
            default_temperature: 기본 온도 설정
            max_retries: 최대 재시도 횟수
        """
        self.provider_name = provider_name.lower()
        self.default_temperature = default_temperature
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)
        
        # API 키 확인 및 클라이언트 초기화
        if self.provider_name == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
            
            model_name = model_name or os.getenv("LLM_MODEL_NAME", "gpt-4o")
            self._client = OpenAIClient(api_key, model_name, default_temperature, max_retries)
        else:
            raise ValueError(f"지원하지 않는 LLM 제공자: {provider_name}. 'openai'만 지원됩니다.")
        
        self.logger.info(f"LLM 클라이언트 초기화 완료 - 제공자: {provider_name}, 모델: {model_name}")

    def generate_text(self, prompt: str, system_prompt: str = None, 
                     temperature: float = None, stop_sequences: List[str] = None) -> str:
        """텍스트 생성"""
        return self._client.generate_response(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=4000
        )["content"]
    
    def generate_chat_completion(self, messages: List[Dict], 
                               temperature: float = None, stop_sequences: List[str] = None) -> str:
        """채팅 형식 API 호출"""
        return self._client.generate_response(
            messages,
            temperature=temperature,
            max_tokens=4000
        )["content"]
    
    def is_available(self) -> bool:
        """API 사용 가능 여부 확인"""
        try:
            # 간단한 테스트 프롬프트로 연결 확인
            test_response = self.generate_text(
                "테스트", 
                temperature=0.1,
                stop_sequences=None
            )
            return bool(test_response and len(test_response.strip()) > 0)
        except Exception as e:
            self.logger.error(f"LLM API 연결 테스트 실패: {e}")
            return False
    
    def test_connection(self) -> dict:
        """
        API 연결 상태를 자세히 테스트
        
        Returns:
            dict: 테스트 결과 정보
        """
        test_result = {
            'available': False,
            'provider': self.provider_name,
            'model': getattr(self._client, 'model_name', 'unknown'),
            'error': None,
            'response_time': None,
            'test_response': None
        }
        
        try:
            import time
            start_time = time.time()
            
            # 테스트 프롬프트 실행
            test_prompt = "Hello, this is a connection test. Please respond with 'OK'."
            response = self.generate_text(test_prompt, temperature=0.1)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            test_result.update({
                'available': True,
                'response_time': round(response_time, 2),
                'test_response': response[:100] if response else None
            })
            
        except Exception as e:
            test_result['error'] = str(e)
            self.logger.error(f"API 연결 테스트 실패: {e}")
        
        return test_result
    
    def get_client_info(self) -> dict:
        """클라이언트 정보 반환"""
        return {
            'provider': self.provider_name,
            'model': getattr(self._client, 'model_name', 'unknown'),
            'temperature': getattr(self._client, 'default_temperature', 0.7),
            'max_retries': getattr(self._client, 'max_retries', 3)
        } 