#  Text-to-Statistical-Test: 최종 아키텍처 설계도

## 1. 프로젝트 비전 및 핵심 아키텍처

### 1.1. 비전

본 프로젝트는 사용자가 자연어로 던진 질문 하나만으로, AI Agent가 데이터의 특성을 파악하고, 통계학적 절차에 따라 스스로 최적의 분석을 수행하며, 최종적으로 비즈니스 관점의 인사이트가 담긴 보고서를 생성하는 **완전 자율형 통계 분석 시스템**을 구축하는 것을 목표로 한다.

### 1.2. 핵심 아키텍처: `Orchestrator-Engine` 모델

이 시스템의 아키텍처는 **'지휘자(Orchestrator)'**와 **'엔진(Engine)'**이라는 두 가지 핵심 개념으로 분리된다.

- **지휘자 (`core/`)**: 전체 분석의 흐름(파이프라인)을 관리하고, 각 단계에서 어떤 작업이 필요한지를 결정한다. 하지만 실제 복잡한 연산이나 추론은 직접 수행하지 않고, 각 단계에 맞는 엔진에게 작업을 위임한다.
- **엔진 (`services/`)**: 지휘자의 요청을 받아 실제적인 작업(LLM 추론, 통계 계산, RAG 검색 등)을 수행하는 강력한 전문 서비스 모음이다.

이러한 분리 구조는 파이프라인을 단순하게 유지하면서도, 각 서비스의 전문성과 재사용성을 극대화하여 유연하고 확장 가능한 시스템을 만든다.

## 2. 데이터 흐름 및 3단계 워크플로우

시스템의 모든 데이터는 `core.context.AppContext` 객체를 통해 파이프라인 단계 간에 전달된다. 실행은 `Orchestrator`에 의해 시작되며, 다음과 같은 3단계 파이프라인을 순차적으로 수행한다.

1.  **데이터 로드 (`DataSelectionStep`)**: 사용자가 지정한 `file_path`를 받아 데이터를 로드하고, `dataframe`을 `AppContext`에 저장한다.
2.  **자율 분석 (`AutonomousAnalysisStep`)**: `dataframe`과 `user_request`를 받아, `AutonomousAgent`를 통해 분석을 수행한다. 이 과정에서 `analysis_plan`, `execution_results`, `final_summary` 등 분석의 모든 결과물을 `AppContext`에 저장한다.
3.  **보고서 생성 (`ReportingStep`)**: `AutonomousAnalysisStep`이 저장한 결과물들을 종합하여 최종 보고서(Markdown 문자열)를 생성하고, 이를 `AppContext`에 저장한다. 최종 결과는 CLI에 출력되고 파일로도 저장된다.

---

## 3. 계층별 상세 설계

### 3.1. `core/` 계층: 지휘자

`core` 계층은 전체 워크플로우를 조율하고, `services` 계층의 기능들을 조합하여 비즈니스 로직을 완성한다.

- **`core.workflow.orchestrator.Orchestrator`**:
    - **역할**: 3단계 파이프라인의 최고 지휘자. `run` 메서드를 통해 각 `PipelineStep`을 순서대로 실행하고, `AppContext`를 전달하여 데이터 흐름을 관리한다.
    - **주요 기능**: `importlib`를 사용해 각 단계를 동적으로 로드함으로써, 파이프라인의 유연성을 확보한다.

- **`core.pipeline.*_step.py`**:
    - **역할**: 파이프라인의 각 단계를 나타내는 '접착제' 코드. `AppContext`에서 필요한 데이터를 꺼내 `Service`를 호출하고, 그 결과를 다시 `AppContext`에 저장한다.
    - **종류**: `DataSelectionStep`, `AutonomousAnalysisStep`, `ReportingStep`.

- **`core.agent.autonomous_agent.AutonomousAgent`**:
    - **역할**: 자율 분석의 두뇌. **[계획 수립 → 도구 실행 → 결과 종합]**의 3단계 내부 워크플로우를 통해 통계 분석을 자율적으로 수행한다.
    - **주요 기능**:
        - `_create_analysis_plan`: `LLMService`를 호출하여 상세한 단계별 분석 계획을 수립한다.
        - `_execute_plan`: 수립된 계획에 따라 `ToolRegistry`의 도구들을 순차적으로 실행한다. 이 과정에서 Agent의 핵심 책임인 **실행 컨텍스트 관리(Execution Context Management)**를 수행한다.
            - 후속 분석 단계(예: 효과 크기, 사후 분석)는 선행 분석 단계의 **결과**뿐만 아니라, 분석에 사용된 **원본 파라미터**에도 의존한다.
            - 따라서 `_execute_plan`은 주요 분석(`run_statistical_test`)이 실행될 때, 해당 단계의 **결과와 파라미터를 모두** 내부 컨텍스트 변수에 저장한다.
            - 후속 단계 실행 시, Agent는 이 저장된 컨텍스트(파라미터와 결과)를 다음 도구의 입력으로 **안정적으로 주입(inject)**한다. 이 메커니즘은 LLM 계획의 불완전성을 보완하고 시스템 전체의 안정성을 보장한다.
        - `_interpret_results`: 모든 실행 결과를 `LLMService`에 전달하여 최종 요약을 생성한다.
        - `run_analysis`: 위 세 가지 기능을 조율하고, 분석의 모든 과정을 담은 포괄적인 딕셔너리를 반환한다. 반환되는 딕셔너리는 `analysis_plan`, `execution_results`, `final_summary` 키를 포함한다.
            - `execution_results`는 각 분석 단계의 결과를 담는 리스트이며, 각 원소는 `step_name`, `tool_name`, `params`, `output`, `status` 키를 가진 딕셔너리이다.
            - **`final_summary`는 `key_findings`, `conclusion` 키를 반드시 포함하는 딕셔너리 형태여야 하며, 이는 리포팅 단계와의 규약이다.**

- **`core.agent.tools.ToolRegistry`**:
    - **역할**: `AutonomousAgent`가 사용할 수 있는 도구(Tool)들을 정의하고 관리하는 레지스트리.
    - **주요 기능**: `StatisticsService`의 실제 함수들을 Agent가 호출하기 쉬운 형태로 감싸고, LLM이 이해할 수 있는 명세(Tool Definition)를 제공한다. 각 도구의 설명은 LLM이 올바른 분석 계획을 수립하도록 명확하고 구체적으로 작성되어야 한다.
        - `run_statistical_test`: t-test, ANOVA(일원/이원), 선형/로지스틱 회귀, 상관 분석, 카이제곱(독립성/적합도), 비율 검정 등 핵심 통계 검정을 수행한다. 선형 회귀 분석의 경우, 잔차 정규성/등분산성/다중공선성 등 관련 가정 검토를 내부적으로 모두 수행하여 결과에 포함하므로, `check_assumption`을 별도로 호출할 필요가 없다.
        - `check_assumption`: **t-test, ANOVA와 같은 그룹 간 평균 비교 분석**을 수행하기 전에, 데이터가 해당 분석의 통계적 가정을 충족하는지(예: 정규성, 등분산성) 검증한다. 회귀 분석이나 상관 분석에는 사용하지 않는다.
        - `calculate_effect_size`: 통계적 유의성(p-value)을 넘어, 발견된 차이나 관계의 **실질적인 크기**(예: Cohen's d, Eta-squared, 상관계수 r, Odds Ratio)를 정량화한다. 선형/로지스틱 회귀 분석의 경우, R-제곱 또는 Pseudo R-제곱 값이 이미 효과 크기의 역할을 하므로 이 도구를 호출할 필요가 없다.
        - `run_posthoc_test`: ANOVA 분석 결과 그룹 간에 유의미한 차이가 발견되었을 때, 구체적으로 어떤 그룹들 사이에 차이가 있는지를 식별하기 위해 사용된다.

- **`core.context.AppContext`**:
    - **역할**: 파이프라인 전반에 걸쳐 공유되는 중앙 데이터 저장소. `dict`를 상속받아 구현되었으며, 워크플로우의 모든 상태(데이터, 중간 결과, 최종 결과 등)를 담는다.

### 3.2. `services/` 계층: 엔진

`services` 계층은 특정 도메인의 작업을 수행하는 독립적인 서비스들의 모음이다. `core` 계층에 의존하지 않는다.

- **`services.llm.LLMService`**:
    - **역할**: LLM과 관련된 모든 기능을 제공하는 퍼사드(Facade).
    - **주요 기능**: 프롬프트 생성, OpenAI API 호출, 응답 파싱(JSON 추출) 등 LLM을 사용하는 데 필요한 모든 복잡한 작업을 캡슐화한다.

- **`services.rag.RAGService`**:
    - **역할**: RAG(검색 증강 생성) 기능을 제공.
    - **주요 기능**: `knowledge_base`의 문서를 벡터화하여 `lancedb`에 저장하고, 주어진 쿼리에 대해 가장 관련성 높은 문서를 검색하여 컨텍스트를 제공한다.

- **`services.statistics.StatisticsService`**:
    - **역할**: 모든 통계 분석 기능을 제공하는 퍼사드.
    - **주요 기능**: `execute_test` 메서드를 통해 `test_id`에 따라 내부 통계 함수를 지능적으로 호출하며, 이때 함수의 시그니처 유형을 구분하여 처리한다.
        - **호출 방식 분기**: `execute_test`는 내부적으로 함수 유형을 **데이터프레임 기반** (예: `run_independent_t_test`)과 **파라미터 기반** (예: `run_two_proportion_test`)으로 구분한다. 파라미터 기반 함수를 호출할 때는 데이터프레임을 인자로 전달하지 않아 시그니처 불일치를 방지한다.
        - **어댑터 역할**: `two_proportion_test`와 같이, LLM이 제공하는 상위 수준 파라미터(예: `group_col`)로부터 실제 연산에 필요한 하위 수준 파라미터(예: `count`, `nobs`)를 계산하여 변환하는 역할을 수행한다.
        - **지원 테스트**: 상관 분석(Correlation), 로지스틱 회귀 분석(Logistic Regression), 이원 분산 분석(Two-Way ANOVA)을 포함하여, 선형 회귀 분석, 카이제곱 검정(독립성/적합도), 비율 검정(단일/두 표본) 등 다양한 통계 분석을 지원한다.

- **`services.reporting.ReportService`**:
    - **역할**: 분석 결과를 바탕으로 최종 **Markdown** 보고서를 생성하는 퍼사드.
    - **주요 기능**: `AppContext`에서 전달받은 분석 요약, 통계 결과 등을 조합하여, 비즈니스 맥락이 포함된 자연어 설명과 함께 **완결된 Markdown 문자열**을 생성한다.

- **`services.code_executor.SafeCodeRunner`**:
    - **역할**: LLM이 생성한 코드를 안전하게 실행하는 기능.
    - **주요 기능**: 격리된 환경에서 코드를 실행하고 결과를 반환한다. 세션 ID와 데이터프레임에 따라 상태가 달라지므로, 전역 인스턴스로 관리되지 않고 필요한 시점에 동적으로 생성된다.

### 3.3. `utils/` 및 `config/`

- **`utils/`**: 프로젝트 전반에서 사용되는 순수 유틸리티(에러 핸들러, 커스텀 JSON 인코더 등) 모음. 다른 계층에 의존하지 않는다.
- **`config/`**: 애플리케이션의 모든 설정(경로, API 키, 로그 레벨 등)을 중앙에서 관리한다. `get_settings()` 함수를 통해 어디서든 일관된 설정 값을 참조할 수 있도록 한다.