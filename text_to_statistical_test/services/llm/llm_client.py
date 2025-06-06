"""
LLM Client

다양한 LLM API (OpenAI, Anthropic 등)를 일관된 인터페이스로 사용할 수 있도록 추상화
재시도 로직, 타임아웃, API 키 관리 등을 포함
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod
import os
from dataclasses import dataclass
import json

from utils.error_handler import ErrorHandler, LLMException
from utils.helpers import retry_on_exception
from utils.global_cache import get_global_cache

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """LLM 응답 데이터 클래스"""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    response_time: float
    metadata: Dict[str, Any] = None

class BaseLLMProvider(ABC):
    """LLM 제공자 기본 클래스"""
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """응답 생성"""
        pass
    
    @abstractmethod
    async def generate_response_async(self, prompt: str, **kwargs) -> LLMResponse:
        """비동기 응답 생성"""
        pass

class OpenAIProvider(BaseLLMProvider):
    """OpenAI API 제공자"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = "https://api.openai.com/v1"
        
        if not self.api_key:
            raise LLMException("OpenAI API 키가 설정되지 않았습니다.")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise LLMException("openai 패키지가 설치되지 않았습니다.")
    
    def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """OpenAI API를 통한 응답 생성"""
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=kwargs.get('model', self.model),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 1.0),
                frequency_penalty=kwargs.get('frequency_penalty', 0.0),
                presence_penalty=kwargs.get('presence_penalty', 0.0)
            )
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage=response.usage.model_dump(),
                finish_reason=response.choices[0].finish_reason,
                response_time=response_time,
                metadata={'provider': 'openai'}
            )
            
        except Exception as e:
            raise LLMException(f"OpenAI API 호출 실패: {str(e)}")
    
    async def generate_response_async(self, prompt: str, **kwargs) -> LLMResponse:
        """비동기 OpenAI API 호출"""
        # 동기 호출을 비동기로 래핑
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_response, prompt, **kwargs)

class AnthropicProvider(BaseLLMProvider):
    """Anthropic API 제공자"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise LLMException("Anthropic API 키가 설정되지 않았습니다.")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise LLMException("anthropic 패키지가 설치되지 않았습니다.")
    
    def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Anthropic API를 통한 응답 생성"""
        start_time = time.time()
        
        try:
            response = self.client.messages.create(
                model=kwargs.get('model', self.model),
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.7),
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.content[0].text,
                model=response.model,
                usage={'input_tokens': response.usage.input_tokens, 
                      'output_tokens': response.usage.output_tokens},
                finish_reason=response.stop_reason,
                response_time=response_time,
                metadata={'provider': 'anthropic'}
            )
            
        except Exception as e:
            raise LLMException(f"Anthropic API 호출 실패: {str(e)}")
    
    async def generate_response_async(self, prompt: str, **kwargs) -> LLMResponse:
        """비동기 Anthropic API 호출"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_response, prompt, **kwargs)

class MockProvider(BaseLLMProvider):
    """테스트용 Mock 제공자"""
    
    def __init__(self, model: str = "mock-model"):
        self.model = model
    
    def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Mock 응답 생성"""
        return LLMResponse(
            content=f"Mock response for prompt: {prompt[:50]}...",
            model=self.model,
            usage={'input_tokens': len(prompt.split()), 'output_tokens': 20},
            finish_reason="stop",
            response_time=0.1,
            metadata={'provider': 'mock'}
        )
    
    async def generate_response_async(self, prompt: str, **kwargs) -> LLMResponse:
        """비동기 Mock 응답"""
        await asyncio.sleep(0.1)  # 실제 API 호출 시뮬레이션
        return self.generate_response(prompt, **kwargs)

class LLMClient:
    """LLM 클라이언트 메인 클래스"""
    
    def __init__(self, 
                 provider: str = "openai",
                 model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 max_retries: int = 3,
                 timeout: float = 30.0,
                 cache_enabled: bool = True):
        """
        LLM 클라이언트 초기화
        
        Args:
            provider: LLM 제공자 ("openai", "anthropic", "mock")
            model: 사용할 모델명
            api_key: API 키
            max_retries: 최대 재시도 횟수
            timeout: 타임아웃 (초)
            cache_enabled: 캐시 사용 여부
        """
        self.provider_name = provider
        self.max_retries = max_retries
        self.timeout = timeout
        self.cache_enabled = cache_enabled
        
        if cache_enabled:
            self.cache = get_global_cache()
        
        self.error_handler = ErrorHandler()
        
        # 제공자 초기화
        self.provider = self._initialize_provider(provider, model, api_key)
        
        logger.info(f"LLM 클라이언트 초기화 완료 - 제공자: {provider}")
    
    def _initialize_provider(self, provider: str, model: Optional[str], api_key: Optional[str]) -> BaseLLMProvider:
        """LLM 제공자 초기화"""
        if provider == "openai":
            return OpenAIProvider(api_key=api_key, model=model or "gpt-4")
        elif provider == "anthropic":
            return AnthropicProvider(api_key=api_key, model=model or "claude-3-sonnet-20240229")
        elif provider == "mock":
            return MockProvider(model=model or "mock-model")
        else:
            raise LLMException(f"지원하지 않는 LLM 제공자: {provider}")
    
    @retry_on_exception(max_attempts=3, delay=1.0, exceptions=(LLMException,))
    def generate_response(self, 
                         prompt: str,
                         max_tokens: int = 1000,
                         temperature: float = 0.7,
                         use_cache: bool = True,
                         **kwargs) -> LLMResponse:
        """
        LLM 응답 생성
        
        Args:
            prompt: 입력 프롬프트
            max_tokens: 최대 토큰 수
            temperature: 온도 (창의성)
            use_cache: 캐시 사용 여부
            **kwargs: 추가 파라미터
            
        Returns:
            LLMResponse: LLM 응답
        """
        # 캐시 확인
        if self.cache_enabled and use_cache:
            cache_key = self._generate_cache_key(prompt, max_tokens, temperature, kwargs)
            cached_response = self.cache.get(cache_key)
            if cached_response:
                logger.debug("캐시된 LLM 응답 반환")
                return cached_response
        
        try:
            # LLM 호출
            response = self.provider.generate_response(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            # 캐시 저장
            if self.cache_enabled and use_cache:
                self.cache.set(cache_key, response, ttl=3600)  # 1시간 캐시
            
            logger.debug(f"LLM 응답 생성 완료 - 토큰: {response.usage}")
            return response
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'prompt_length': len(prompt)})
            raise LLMException(f"LLM 응답 생성 실패: {error_info['message']}")
    
    async def generate_response_async(self, 
                                    prompt: str,
                                    max_tokens: int = 1000,
                                    temperature: float = 0.7,
                                    use_cache: bool = True,
                                    **kwargs) -> LLMResponse:
        """비동기 LLM 응답 생성"""
        # 캐시 확인
        if self.cache_enabled and use_cache:
            cache_key = self._generate_cache_key(prompt, max_tokens, temperature, kwargs)
            cached_response = self.cache.get(cache_key)
            if cached_response:
                logger.debug("캐시된 LLM 응답 반환 (비동기)")
                return cached_response
        
        try:
            # 비동기 LLM 호출
            response = await self.provider.generate_response_async(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            # 캐시 저장
            if self.cache_enabled and use_cache:
                self.cache.set(cache_key, response, ttl=3600)
            
            logger.debug(f"비동기 LLM 응답 생성 완료 - 토큰: {response.usage}")
            return response
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context={'prompt_length': len(prompt)})
            raise LLMException(f"비동기 LLM 응답 생성 실패: {error_info['message']}")
    
    def generate_batch_responses(self, 
                               prompts: List[str],
                               **kwargs) -> List[LLMResponse]:
        """배치 응답 생성"""
        responses = []
        for prompt in prompts:
            try:
                response = self.generate_response(prompt, **kwargs)
                responses.append(response)
            except Exception as e:
                logger.error(f"배치 처리 중 오류: {str(e)}")
                # 오류 응답 생성
                error_response = LLMResponse(
                    content=f"Error: {str(e)}",
                    model="error",
                    usage={'input_tokens': 0, 'output_tokens': 0},
                    finish_reason="error",
                    response_time=0.0,
                    metadata={'error': True}
                )
                responses.append(error_response)
        
        return responses
    
    async def generate_batch_responses_async(self, 
                                           prompts: List[str],
                                           **kwargs) -> List[LLMResponse]:
        """비동기 배치 응답 생성"""
        tasks = [
            self.generate_response_async(prompt, **kwargs) 
            for prompt in prompts
        ]
        
        responses = []
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"비동기 배치 처리 중 오류: {str(result)}")
                error_response = LLMResponse(
                    content=f"Error: {str(result)}",
                    model="error",
                    usage={'input_tokens': 0, 'output_tokens': 0},
                    finish_reason="error",
                    response_time=0.0,
                    metadata={'error': True}
                )
                responses.append(error_response)
            else:
                responses.append(result)
        
        return responses
    
    def _generate_cache_key(self, prompt: str, max_tokens: int, temperature: float, kwargs: Dict) -> str:
        """캐시 키 생성"""
        import hashlib
        
        cache_data = {
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'provider': self.provider_name,
            'kwargs': kwargs
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """사용량 통계 조회"""
        if hasattr(self.cache, 'get_stats'):
            cache_stats = self.cache.get_stats()
        else:
            cache_stats = {}
        
        return {
            'provider': self.provider_name,
            'cache_enabled': self.cache_enabled,
            'cache_stats': cache_stats,
            'max_retries': self.max_retries,
            'timeout': self.timeout
        }
    
    def clear_cache(self):
        """캐시 클리어"""
        if self.cache_enabled:
            # LLM 관련 캐시만 클리어 (구현 필요)
            logger.info("LLM 캐시 클리어됨") 