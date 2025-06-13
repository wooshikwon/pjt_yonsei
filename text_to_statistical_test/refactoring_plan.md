# 📊 텍스트-통계테스트 시스템 전체 아키텍처 분석 보고서

## 1. 전체적인 파이프라인 및 구현 상세사항 분석

### 1.1 현재 아키텍처 개요

**핵심 설계 패턴:**
- **Orchestrator Pattern**: `core/pipeline/orchestrator.py`가 전체 워크플로우 관리
- **Facade Pattern**: 각 서비스(`LLMService`, `StatisticsService` 등)가 복잡한 내부 로직을 단순한 인터페이스로 제공
- **Dependency Injection**: 서비스 간 의존성을 생성자 주입으로 관리
- **Singleton Pattern**: `RAGService`, `LLMClient` 등 리소스 집약적 객체들

**현재 데이터 흐름:**
```
User Request → Orchestrator → Step1(데이터선택) → Step2(자율분석) → Step3(보고서생성)
                                    ↓
                            AutonomousAgent (LLMService + StatisticsService)
                                    ↓
                            ToolRegistry (StatisticsService 연동)
```

### 1.2 각 계층별 세부 분석

#### 1.2.1 Core Layer (`core/`)
**Pipeline 모듈:**
- `orchestrator.py`: 전체 워크플로우 지휘자, 모든 서비스 인스턴스 관리
- `step2_autonomous_analysis.py`: LLM과 Statistics 서비스 통합 지점
- `app_context.py`: 단계 간 데이터 전달 컨텍스트

**Agent 모듈:**
- `autonomous_agent.py`: LLM 기반 자율 분석 실행
- `tools.py`: Statistics 서비스를 도구로 래핑하여 LLM에 제공

#### 1.2.2 Services Layer (`services/`)
**LLM 서비스:**
- `llm_service.py`: LLM 관련 모든 기능의 Facade
- `llm_client.py`: OpenAI API 통신 관리 (Singleton)
- `llm_response_parser.py`: LLM 응답 파싱 및 구조화

**Statistics 서비스:**
- `stats_service.py`: 통계 분석 기능 통합 제공
- `dispatchers.py`: 통계 함수들의 라우팅 테이블

#### 1.2.3 Configuration & Utilities
**설정 관리:**
- `config/settings.py`: 중앙집중식 설정 관리
- `config/logging_config.py`: 로깅 시스템 구성

**에러 처리:**
- `utils/error_handler.py`: 표준화된 에러 코드 및 예외 클래스

### 1.3 주요 강점

✅ **모듈화된 설계**: 각 서비스가 명확한 책임을 가짐  
✅ **의존성 주입**: 테스트 가능하고 유연한 구조  
✅ **중앙집중식 설정**: `get_settings()`로 일관된 설정 관리  
✅ **표준화된 에러 처리**: `ErrorCode` 열거형 사용  
✅ **비동기 처리**: 모든 LLM 호출이 async/await 패턴

### 1.4 현재 일관성 문제점

#### 🔴 심각한 문제점

1. **서비스 인스턴스 생성 방식 불일치**
   - `services/__init__.py`에서 모듈 레벨 인스턴스 생성
   - `orchestrator.py`에서 `StatisticsService()` 별도 생성
   - 동일한 서비스의 여러 인스턴스 존재 가능성

2. **의존성 관리 혼재**
   - 일부는 글로벌 인스턴스 참조 (`from services import llm_service`)
   - 일부는 생성자 주입 (`StatisticsService()` 직접 생성)

3. **에러 처리 패턴 불일치**
   - LLM 관련: `LLMException` + `ErrorCode` 조합
   - Statistics 관련: 일반 `ValueError`, `NotImplementedError` 사용
   - Agent 관련: `RuntimeError` 사용

#### 🟡 중간 문제점

4. **로깅 패턴 불일치**
   - 일부 모듈: `self.logger = logging.getLogger(__name__)`
   - 일부 모듈: `logger = logging.getLogger(__name__)` (모듈 레벨)

5. **응답 형식 표준화 부족**
   - Statistics 서비스: `{"assumption_checks": ..., "test_results": ...}`
   - LLM 서비스: 다양한 구조의 딕셔너리 반환

6. **설정 사용 방식 불일치**
   - 대부분: `get_settings()` 사용
   - 일부: 직접 환경변수 참조

## 2. 구체적인 수정 계획

### 2.1 Phase 1: 의존성 관리 통일 (우선순위: 🔴 높음)

#### 2.1.1 서비스 생명주기 관리 표준화

**목표:** 모든 서비스 인스턴스를 단일 지점에서 관리

**수정 방법:**

1. **`services/__init__.py` 개선:**
```python
# 현재 문제: 모듈 레벨에서 즉시 인스턴스 생성
rag_service = RAGService(knowledge_base_dir="knowledge_base")
llm_service = LLMService(rag_service=rag_service)

# 개선안: 팩토리 함수를 통한 지연 초기화
def get_rag_service() -> RAGService:
    if not hasattr(get_rag_service, '_instance'):
        get_rag_service._instance = RAGService(knowledge_base_dir="knowledge_base")
    return get_rag_service._instance

def get_llm_service() -> LLMService:
    if not hasattr(get_llm_service, '_instance'):
        get_llm_service._instance = LLMService(rag_service=get_rag_service())
    return get_llm_service._instance
```

2. **`orchestrator.py` 수정:**
```python
# 현재 문제: 혼재된 인스턴스 생성
from services import llm_service, rag_service, report_service
self.stats_service = StatisticsService()  # 별도 생성

# 개선안: 모든 서비스를 동일한 방식으로 주입
from services import get_llm_service, get_rag_service, get_statistics_service, get_report_service

def __init__(self):
    self.llm_service = get_llm_service()
    self.rag_service = get_rag_service() 
    self.stats_service = get_statistics_service()
    self.report_service = get_report_service()
```

#### 2.1.2 DI Container 도입

**새로운 파일:** `core/di_container.py`
```python
class DIContainer:
    """전역 의존성 관리 컨테이너"""
    _services = {}
    
    @classmethod
    def register_service(cls, service_name: str, factory_func: Callable):
        cls._services[service_name] = factory_func
    
    @classmethod
    def get_service(cls, service_name: str):
        if service_name not in cls._services:
            raise ValueError(f"Service {service_name} not registered")
        return cls._services[service_name]()
```

### 2.2 Phase 2: 에러 처리 통합 (우선순위: 🔴 높음)

#### 2.2.1 통계 서비스 에러 처리 표준화

**`services/statistics/stats_service.py` 수정:**
```python
from utils import StatisticalException
from utils.error_handler import ErrorCode

def execute_test(self, test_id: str, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    if test_id not in self._test_dispatcher:
        raise StatisticalException(
            f"'{test_id}' 통계 검정은 지원되지 않습니다.",
            ErrorCode.STATISTICAL_TEST_NOT_SUPPORTED
        )
```

#### 2.2.2 ErrorCode 확장

**`utils/error_handler.py` 수정:**
```python
class ErrorCode(IntEnum):
    # 기존 코드들...
    STATISTICAL_TEST_NOT_SUPPORTED = 201
    STATISTICAL_ASSUMPTION_FAILED = 202
    STATISTICAL_DATA_INVALID = 203
    AGENT_TOOL_EXECUTION_FAILED = 301
    AGENT_PLAN_GENERATION_FAILED = 302
```

### 2.3 Phase 3: 로깅 패턴 통일 (우선순위: 🟡 중간)

#### 2.3.1 로깅 믹스인 클래스 도입

**새로운 파일:** `utils/logging_mixin.py`
```python
import logging
from typing import Protocol

class LoggingMixin(Protocol):
    """로깅 기능을 제공하는 믹스인"""
    
    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
        return self._logger
```

#### 2.3.2 모든 서비스 클래스에 적용

```python
class LLMService(LoggingMixin):
    def __init__(self, rag_service: RAGService):
        # self.logger는 자동으로 사용 가능
        self.logger.info("LLMService가 초기화되었습니다.")
```

### 2.4 Phase 4: 응답 형식 표준화 (우선순위: 🟡 중간)

#### 2.4.1 표준 응답 모델 정의

**새로운 파일:** `core/response_models.py`
```python
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

@dataclass
class ServiceResponse:
    """모든 서비스의 표준 응답 형식"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_code: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class StatisticalTestResponse(ServiceResponse):
    """통계 테스트 전용 응답 형식"""
    assumption_checks: Optional[Dict[str, Any]] = None
    test_results: Optional[Dict[str, Any]] = None
```

### 2.5 Phase 5: LLM 서비스 향상 (우선순위: 🟢 낮음)

#### 2.5.1 프롬프트 관리 개선

**현재 문제:** 하드코딩된 프롬프트 경로
**개선안:** 프롬프트 레지스트리 패턴

```python
class PromptRegistry:
    """프롬프트 템플릿 중앙 관리"""
    _prompts = {
        'request_structuring': {
            'system': 'request_structuring/system.prompt',
            'human': 'request_structuring/human.prompt'
        }
    }
    
    @classmethod
    def get_prompt_path(cls, category: str, prompt_type: str) -> str:
        return cls._prompts[category][prompt_type]
```

#### 2.5.2 LLM 응답 검증 강화

```python
def parse_json_response(self, response_text: str, response_format: Dict[str, Any]) -> Dict[str, Any]:
    parsed_data = json.loads(json_str)
    
    # JSON 스키마 검증 추가
    try:
        validate(parsed_data, response_format)
        return parsed_data
    except ValidationError as e:
        raise ParsingException(f"응답이 예상 스키마와 일치하지 않습니다: {e}", ErrorCode.VALIDATION_ERROR)
```

### 2.6 구현 우선순위 및 타임라인

**Week 1-2: Phase 1 (의존성 관리)**
- DI Container 구현
- 서비스 팩토리 함수 도입
- Orchestrator 리팩토링

**Week 3: Phase 2 (에러 처리)**
- ErrorCode 확장
- 통계 서비스 에러 처리 통합
- Agent 에러 처리 표준화

**Week 4: Phase 3 (로깅 통일)**
- LoggingMixin 구현
- 모든 서비스 클래스 적용

**Week 5-6: Phase 4 (응답 표준화)**
- 표준 응답 모델 정의
- 모든 서비스 응답 형식 통일

**Week 7: Phase 5 (LLM 서비스 향상)**
- 프롬프트 레지스트리 구현
- 응답 검증 강화

### 2.7 테스트 전략

각 Phase별로 다음 테스트를 수행:

1. **단위 테스트**: 각 서비스의 독립적 기능 검증
2. **통합 테스트**: 서비스 간 연동 검증
3. **E2E 테스트**: 전체 파이프라인 동작 검증
4. **회귀 테스트**: 기존 기능 보존 확인

이러한 체계적인 개선을 통해 코드의 일관성, 유지보수성, 확장성을 크게 향상시킬 수 있습니다.
