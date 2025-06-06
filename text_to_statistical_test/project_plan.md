## Text-to-Statistical-Test 프로젝트 재구성 계획서 (개선안)

### 0. 요약 및 핵심 목표

본 문서는 기존 Text-to-Statistical-Test 프로젝트의 문제점을 분석하고, **RAG(Retrieval Augmented Generation)와 Agentic LLM(Large Language Model)을 활용한 대화형 Multi-turn 자동 통계검정 시스템**으로 재구성하기 위한 구체적인 계획을 제시합니다. 주요 목표는 사용자와의 자연스러운 대화를 통해 데이터 분석 요구사항을 파악하고, RAG를 통해 확보된 도메인 지식 및 통계적 지식을 바탕으로 Agentic LLM이 자율적으로 최적의 통계 분석을 수행하며, 그 결과를 해석하여 사용자에게 제공하는 것입니다. 이 재구성을 통해 시스템의 **모듈성, 확장성, 유지보수성, 그리고 가장 중요하게는 분석 과정의 지능성과 자율성을 대폭 향상**시키고자 합니다.

---

## 1. 워크플로우 의도

본 프로젝트는 **향상된 RAG 기반 Agentic AI 통계 분석 시스템**으로, 사용자가 자연어로 분석 요청을 입력하면, AI가 RAG를 통해 확보한 비즈니스 도메인 지식, DB 스키마 구조, 통계 방법론 지식을 활용하여 최적의 통계 분석 방법을 **자율적으로 계획하고 수행**하는 8단계 워크플로우를 구현합니다. 핵심은 사용자와의 **Multi-turn 대화**를 통해 분석의 정확도를 높이고, Agentic LLM이 분석 전 과정을 주도적으로 이끌어가는 것입니다.

### 재구성된 8단계 워크플로우
1.  **데이터 파일 선택 및 초기 이해**: `input_data/data_files/` 폴더의 데이터 파일 목록을 표시하고, 사용자가 선택합니다. 시스템은 선택된 데이터의 기본적인 메타정보를 파악합니다.
2.  **사용자의 자연어 요청 및 목표 정의 (Multi-turn)**: 사용자가 자연어로 분석 목표와 궁금증을 전달합니다. 시스템은 **대화형으로 추가 질문**을 통해 분석의 범위와 구체적인 목표를 명확히 합니다. 이 과정은 `core/pipeline/user_request.py`와 연계되어 Agentic LLM의 자연어 이해 능력을 활용합니다.
3.  **데이터 심층 분석 및 요약**: 선택된 데이터에 대한 기술 통계, 변수 분포, 잠재적 이슈 (결측치, 이상치 등)를 심층적으로 분석하고 요약하여 사용자에게 제공합니다. 이 과정은 `core/pipeline/data_summary.py`에서 처리하며, Agentic LLM이 RAG를 통해 얻은 데이터 특성 이해를 바탕으로 진행합니다.
4.  **Agentic LLM의 분석 전략 제안 (RAG 활용)**: Agentic LLM (`core/agent/autonomous_agent.py`)은 사용자의 요청, 데이터 특성, RAG를 통해 확보한 **도메인 지식**(예: `resources/knowledge_base/business_domains/`) 및 **통계적 지식**(예: `resources/knowledge_base/statistical_concepts/`)을 종합하여 가능한 분석 방법들과 각 방법의 장단점을 제시합니다. 이 단계는 `core/pipeline/analysis_proposal.py`를 통해 이루어집니다.
5.  **사용자 피드백 기반 분석 방식 구체화 (Multi-turn)**: 사용자는 LLM의 제안을 검토하고, 필요한 경우 추가적인 요구사항이나 선호하는 분석 방향을 제시합니다. 시스템은 이를 반영하여 최종 분석 계획을 확정합니다. 이 단계는 `core/pipeline/user_selection.py`를 통해 사용자와의 상호작용을 관리합니다.
6.  **RAG를 활용한 Agentic LLM의 데이터 분석 계획 수립**: 확정된 분석 목표에 따라 Agentic LLM은 RAG 시스템(`core/rag/`)을 활용하여 필요한 **통계 코드 템플릿** (`resources/knowledge_base/code_templates/`) 및 **DB 스키마 정보** (`input_data/metadata/database_schemas/`)를 참조하여 구체적인 통계 분석 실행 계획을 수립합니다. 이는 `core/pipeline/agent_analysis.py`의 핵심 기능입니다.
7.  **Agentic LLM의 자율적 통계 검정 및 동적 조정 (Agentic Flow)**: `core/agent/flow_controller.py` 와 `core/agent/decision_tree.py` 의 제어를 받는 `AutonomousAgent`가 통계 검정 전 과정을 **자율적으로 수행**합니다.
    * **전제조건 검증**: 정규성, 등분산성 등의 가정을 `services/statistics/tests/assumptions.py`를 활용하여 검증합니다.
    * **동적 분석 방법 조정**: 가정 검증 결과에 따라, 필요한 경우 분석 방법을 **동적으로 수정**합니다. (예: 정규성 불만족 시 비모수 검정으로 자동 전환)
    * **핵심 통계 검정**: 선택된 통계 기법(예: t-검정, ANOVA, 회귀분석 등 `services/statistics/tests/parametric.py` 또는 `nonparametric.py` 활용)을 실행합니다.
    * **사후 분석**: 필요한 경우 (예: ANOVA 후) 사후 검정을 수행합니다.
    * **오류 처리 및 재시도**: 코드 실행 오류나 예상치 못한 데이터 문제 발생 시, Agent는 문제를 진단하고, 가능한 해결책을 시도하거나 사용자에게 명확한 설명을 제공합니다. 이는 `utils/error_handlers.py`와 연동됩니다.
    이 모든 과정은 `core/pipeline/agent_testing.py`를 통해 조율되며, **Agentic Flow의 핵심**을 보여줍니다.
8.  **Agentic LLM의 보고서 생성 및 해석**: 수행된 통계 검정 결과, 시각화 자료(`core/reporting/visualization.py`), 그리고 RAG를 통해 얻은 비즈니스 컨텍스트를 종합하여 사용자 친화적인 **결론 및 비즈니스 인사이트가 포함된 보고서**(`core/reporting/report_builder.py`)를 생성하여 제공합니다. 사용자는 보고서 내용에 대해 추가 질문을 할 수 있으며, LLM은 이에 답변합니다. 이 단계는 `core/pipeline/agent_reporting.py`에서 담당합니다.

### RAG 시스템의 핵심 역할 (재정의)
RAG 시스템 (`core/rag/`)은 Agentic LLM의 분석 품질과 자율성을 극대화하는 데 핵심적인 역할을 수행합니다.
* **지식 저장소 (`knowledge_store.py`)**:
    * **비즈니스 도메인 지식**: `resources/knowledge_base/business_domains/` 내 산업별/업무별 용어, KPI, 일반적인 분석 패턴 등을 포함.
    * **DB 스키마 구조 및 메타데이터**: `input_data/metadata/database_schemas/` 내 테이블 정의, 컬럼 설명, 관계 등을 포함.
    * **통계 개념 및 방법론**: `resources/knowledge_base/statistical_concepts/` 내 각 통계 기법의 가정, 해석 방법, 장단점 등을 포함.
    * **통계 코드 템플릿**: `resources/knowledge_base/code_templates/` 에 파이썬 기반의 통계 분석 코드 조각들을 기법별로 분류하여 저장. Agent가 이를 참조하여 분석 코드 생성/수정.
    * **워크플로우 가이드라인**: `resources/workflow_spec.json` 에 정의된 일반적인 분석 흐름 및 단계별 고려사항. 이는 Agent의 자율성을 제약하는 규칙이 아닌, 참조 가능한 가이드라인으로 활용.
* **쿼리 엔진 (`query_engine.py`)**: 사용자의 질문이나 분석 맥락에 따라 가장 관련성 높은 정보를 지식 저장소에서 효율적으로 검색 (캐싱 기능 포함).
* **컨텍스트 빌더 (`context_builder.py`)**: 검색된 정보를 LLM 프롬프트에 효과적으로 통합하여 Agent의 추론 능력을 향상.

### Agentic Flow의 핵심 개념 (재정의)
Agentic LLM (`core/agent/autonomous_agent.py`)은 단순한 코드 실행기가 아닌, 분석 목표 달성을 위해 자율적으로 판단하고 행동하는 주체입니다.
* **자율적 의사결정**: `core/agent/decision_tree.py` 와 `flow_controller.py`를 기반으로, 통계 분석의 각 단계(가정 검토, 분석 방법 선택, 결과 해석 등)에서 사용자의 개입을 최소화하며 최적의 경로를 스스로 결정합니다.
* **동적 적응**: 데이터의 실제 특성(예: 분포, 크기)이나 가정 검정 결과에 따라, 사전에 정의된 워크플로우나 코드 템플릿에만 의존하지 않고 분석 계획을 유연하게 수정하고 대안적 방법을 탐색합니다.
* **연쇄적 추론 및 학습**: 각 분석 단계의 결과를 다음 단계의 입력으로 활용하며, 전체 분석 과정을 통해 문제 해결 전략을 개선합니다. 오류 발생 시, 원인을 추론하고 해결 방안을 모색하거나 사용자에게 명확한 가이드를 요청합니다.
* **도구 사용 (Tool Usage)**: RAG 시스템, 통계 함수 라이브러리 (`services/statistics/`), 데이터 처리 유틸리티 (`utils/data_loader.py`) 등을 "도구"로서 활용하여 분석 목표를 달성합니다.

---

## 2. 시스템이 구현해야 하는 통계분석 범위

(기존 내용과 동일하나, 각 분석이 Agentic LLM에 의해 어떻게 선택되고 실행될 수 있는지의 관점이 추가됩니다. Agent는 RAG를 통해 각 분석법의 적합성을 판단하고, 필요한 전제조건 검증 및 사후 분석을 자율적으로 수행합니다.)

### 그룹 비교 분석
* **독립표본 t-검정**: 두 독립 그룹 간 평균 비교
* **대응표본 t-검정**: 단일 그룹의 처치 전후 평균 비교
* **일원분산분석 (ANOVA)**: 셋 이상 독립 그룹 간 평균 비교 (사후 분석 포함)
* **이원분산분석 (Two-way ANOVA)**: 두 독립변수가 종속변수에 미치는 영향 및 상호작용 효과 분석
* **Mann-Whitney U 검정**: 두 독립 그룹 간 비모수 비교 (t-검정의 정규성 가정 불만족 시)
* **Kruskal-Wallis 검정**: 셋 이상 독립 그룹 간 비모수 비교 (ANOVA의 정규성 가정 불만족 시)

### 관계 및 상관관계 분석
* **피어슨 상관분석**: 두 연속형 변수 간 선형 관계 강도 측정
* **스피어만 상관분석**: 두 변수 간 순위 기반 단조 관계 측정 (비선형 관계나 이상치에 덜 민감)
* **단순선형회귀분석**: 하나의 독립변수를 사용하여 종속변수를 예측하는 모델링
* **다중선형회귀분석**: 둘 이상의 독립변수를 사용하여 종속변수를 예측하는 모델링 (변수 선택, 다중공선성 진단 포함)
* **로지스틱 회귀분석**: 이진형 또는 범주형 종속변수를 예측하는 모델링

### 범주형 데이터 분석
* **카이제곱 독립성 검정**: 두 범주형 변수 간의 연관성 검정
* **Fisher의 정확검정**: 표본 크기가 작은 경우의 두 범주형 변수 간 연관성 검정
* **McNemar 검정**: 대응되는 두 범주형 변수 간의 비율 변화 검정

### 전제조건 검증, 사후 분석 및 효과 크기
* **정규성 검정**: Shapiro-Wilk test, Kolmogorov-Smirnov test (Agent가 데이터 특성에 따라 선택)
* **등분산성 검정**: Levene's test, Bartlett's test (Agent가 데이터 특성에 따라 선택)
* **사후 검정 (Post-hoc tests)**: Tukey HSD, Bonferroni correction 등 (ANOVA 후 Agent가 필요성 판단 및 수행)
* **효과 크기 (Effect size) 계산**: Cohen's d, Eta-squared, Cramer's V 등 (Agent가 분석 결과의 실제적 중요성 제시)

---

## 3. 새로운 파이프라인 디렉토리 구조 (실제 구현된 구조)

제안된 디렉토리 구조는 모듈성, 책임 분리, 확장성을 극대화하는 데 초점을 맞춥니다.

```
text_to_statistical_test/
├── main.py                      # 진입점 (CLI 또는 API 라우팅, 사용자 세션 관리)
├── pyproject.toml               # Poetry 의존성 및 프로젝트 메타데이터
├── poetry.lock                  # 의존성 잠금 파일
├── .python-version              # Python 버전 명시
├── .env                         # 환경 변수 (기존 env_example.txt 대체)
├── Dockerfile                   # Docker 이미지 빌드 설정
├── docker-compose.yml           # Docker 서비스 정의
├── setup_project.py             # (필요시) 프로젝트 초기 설정 스크립트 (간소화 또는 제거 고려)
├── requirements.txt             # (Poetry 사용 시 자동 생성/관리되므로 필수 아님)
├── .dockerignore                # Docker 빌드 시 제외할 파일 목록
├── input_data/                  # 입력 데이터 및 관련 메타데이터
│   ├── data_files/              # 분석 대상 데이터 파일 (CSV, Excel 등)
│   └── metadata/                # 데이터에 대한 추가 정보 (RAG가 활용)
│       ├── database_schemas/    # (선택적) DB 스키마 JSON 파일들 (테이블, 컬럼, 관계)
│       └── data_dictionaries/   # (선택적) 각 데이터 파일의 변수 설명 JSON 파일들
├── output_data/                 # 분석 결과 저장 (기존 output_results/ 에서 명칭 변경 및 구조화)
│   ├── reports/                 # 생성된 보고서 (HTML, PDF, Markdown 등)
│   ├── visualizations/          # 생성된 시각화 이미지 (PNG, SVG 등)
│   └── analysis_cache/          # 중간 분석 결과, RAG 검색 결과 등 캐시 데이터
├── logs/                        # 애플리케이션 로그 파일
│   └── app.log
├── config/                      # 애플리케이션 설정
│   ├── settings.py              # 통합 설정 (API 키, 기본 경로, LLM 모델 등)
│   ├── logging_config.py        # 로깅 레벨, 포맷, 핸들러 등 상세 설정
│   └── error_codes.py           # (선택적) 표준화된 오류 코드 및 메시지 정의
├── core/                        # 핵심 로직: 파이프라인, RAG, Agent, 워크플로우, 보고
│   ├── __init__.py
│   ├── pipeline/                # 8단계 워크플로우 각 단계를 처리하는 모듈
│   │   ├── __init__.py
│   │   ├── base_pipeline_step.py # 파이프라인 단계의 기본 클래스
│   │   ├── data_selection.py     # 1단계: 데이터 선택 및 초기 이해
│   │   ├── user_request.py       # 2단계: 사용자 자연어 요청 및 목표 정의
│   │   ├── data_summary.py       # 3단계: 데이터 심층 분석 및 요약
│   │   ├── analysis_proposal.py  # 4단계: Agentic LLM의 분석 전략 제안
│   │   ├── user_selection.py     # 5단계: 사용자 피드백 기반 분석 방식 구체화
│   │   ├── agent_analysis.py     # 6단계: RAG 기반 LLM Agent 데이터 분석 계획 수립
│   │   ├── agent_execution.py    # 7단계: Agentic LLM의 자율적 통계 검정
│   │   └── agent_reporting.py    # 8단계: Agentic LLM의 보고서 생성 및 해석
│   ├── rag/                     # RAG 시스템 (독립성 및 기능 강화)
│   │   ├── __init__.py
│   │   ├── rag_manager.py       # RAG 시스템 통합 관리자 (신규 추가)
│   │   ├── knowledge_store.py   # 지식 저장소 관리 (벡터DB 연동, 문서 로딩/파싱)
│   │   ├── query_engine.py      # 검색 및 랭킹 로직
│   │   ├── context_builder.py   # LLM 프롬프트용 컨텍스트 생성
│   │   ├── vector_store.py      # 벡터 저장소 관리
│   │   ├── retriever.py         # 문서 검색 및 랭킹
│   │   └── rag_cache_manager.py # RAG 검색 결과 캐싱
│   ├── agent/                   # Agentic LLM 시스템 (자율성, 동적 적응)
│   │   ├── __init__.py
│   │   ├── autonomous_agent.py  # 자율적 의사결정 및 행동 주체 LLM Agent
│   │   ├── flow_controller.py   # Agentic Flow 제어 (상태, 전환, 도구 사용 관리)
│   │   ├── decision_tree.py     # 동적 의사결정 로직 (통계 검정 경로 탐색)
│   │   └── tool_registry.py     # Agent가 사용할 수 있는 도구(함수) 등록 및 관리
│   ├── workflow/                # 전체 워크플로우 오케스트레이션 및 상태 관리
│   │   ├── __init__.py
│   │   ├── orchestrator.py      # 8단계 파이프라인의 순차적/조건부 실행 관리
│   │   ├── state_manager.py     # 대화 상태, 분석 진행 상태 등 관리
│   │   └── conversation_history.py # 대화 이력 관리
│   └── reporting/               # 결과 보고 및 시각화
│       ├── __init__.py
│       ├── report_builder.py    # 다양한 형식의 보고서 생성 로직
│       ├── visualization_engine.py # 데이터 시각화 생성 (Plotly, Matplotlib 등 연동)
│       └── output_formatter.py  # 결과 텍스트, 테이블 등의 포맷팅
├── services/                    # 외부 서비스 연동 및 핵심 비즈니스 로직 (LLM, 통계)
│   ├── __init__.py
│   ├── llm/                     # LLM 관련 서비스
│   │   ├── __init__.py
│   │   ├── llm_client.py        # LLM API 클라이언트 (OpenAI, Anthropic 등)
│   │   ├── prompt_engine.py     # 동적 프롬프트 생성 및 관리
│   │   └── llm_response_parser.py # LLM 응답 파싱 및 검증
│   ├── statistics/              # 통계 분석 로직
│   │   ├── __init__.py
│   │   ├── stats_executor.py    # 통계 분석 실행 통합 관리자
│   │   ├── data_preprocessor.py # 데이터 전처리 (결측치, 이상치, 스케일링 등)
│   │   ├── descriptive_stats.py # 기술 통계 계산
│   │   ├── inferential_tests/   # 추론 통계 검정 모듈 (하위 디렉토리로 구조화)
│   │   │   ├── __init__.py
│   │   │   ├── assumption_checks.py # 정규성, 등분산성 등 가정 검정
│   │   │   ├── parametric_tests.py  # t-검정, ANOVA 등
│   │   │   ├── nonparametric_tests.py # Mann-Whitney U, Kruskal-Wallis 등
│   │   │   ├── regression_analysis.py # 선형, 로지스틱 회귀 등
│   │   │   └── categorical_analysis.py # 카이제곱, McNemar 등
│   │   └── posthoc_analysis.py  # 사후 분석
│   ├── code_executor/           # LLM이 생성하거나 참조한 코드의 안전한 실행 환경
│   │   ├── __init__.py
│   │   └── safe_code_runner.py  # 코드 실행 (샌드박스 환경 고려)
│   └── visualization/           # 시각화 서비스 (신규 추가)
│       ├── __init__.py
│       └── chart_generator.py   # 차트 생성 및 관리
├── utils/                       # 범용 유틸리티 함수 및 클래스 (성능 최적화 완료)
│   ├── __init__.py
│   ├── data_loader.py           # 고성능 데이터 로딩 (청크 처리, 메모리 최적화, 성능 메트릭)
│   ├── data_utils.py            # 파일 시스템 기반 데이터 유틸리티 (역할 재정의)
│   ├── input_validator.py       # 사용자 입력 및 데이터 유효성 검사
│   ├── global_cache.py          # 고성능 캐싱 시스템 (TTL 최적화, 메트릭 추적, 건강 상태 모니터링)
│   ├── error_handler.py         # 표준화된 오류 처리 및 예외 정의
│   ├── helpers.py               # 중앙화된 헬퍼 함수들 (중복 제거 완료)
│   └── ui_helpers.py            # 사용자 인터페이스 관련 유틸리티
└── resources/                   # RAG 시스템의 지식 베이스 및 기타 정적 리소스
    ├── __init__.py
    ├── workflow_spec.json       # 워크플로우 단계별 가이드라인 및 기본 로직 명세
    ├── statistical_methods.json # 각 통계 기법 메타데이터 (이름, 설명, 사용 조건 등 RAG가 참조)
    └── knowledge_base/          # RAG가 학습하고 참조하는 핵심 지식 데이터
        ├── business_domains/    # 산업/도메인별 전문 지식 (Markdown, JSON 등)
        │   ├── healthcare.md
        │   └── finance.json
        ├── statistical_concepts/ # 통계학적 개념, 공식, 해석 방법 (Markdown, JSON 등)
        │   ├── hypothesis_testing.md
        │   └── regression_deep_dive.md
        └── code_templates/      # 통계 분석 Python 코드 템플릿 (기법별 .py 파일 또는 JSON)
            ├── python/
            │   ├── t_test_template.py
            │   └── anova_template.py
```

### Utils 모듈 최적화 완료 사항

**Phase 4-5에서 완료된 Utils 성능 최적화:**

1. **data_loader.py**: 
   - 대용량 파일 청크 단위 처리 (`load_file_chunked`, `_load_large_csv`, `_load_large_excel`)
   - 메모리 사용량 모니터링 및 자동 데이터 타입 최적화
   - 로딩 성능 메트릭 추적 (`LoadingMetrics` 클래스)
   - `@cached` 데코레이터를 통한 global_cache 통합

2. **global_cache.py**:
   - 캐시 성능 메트릭 시스템 (`CacheMetrics`, `CacheEntry` 클래스)
   - TTL 자동 최적화 (`optimize_ttl()` 메서드)
   - 캐시 건강 상태 모니터링 (`get_cache_health()`)
   - LRU 기반 메모리 관리 및 자동 정리

3. **data_utils.py**:
   - 역할 재정의: Pandas 의존성 제거, 순수 파일 시스템 유틸리티로 축소
   - `validate_data_file` → `validate_file_access`로 변경 (InputValidator 활용)
   - 새로운 함수들: `get_file_basic_info`, `compare_data_files`, `create_data_directory_structure`

4. **helpers.py**:
   - 중복 함수 중앙화: `detect_csv_delimiter`, `get_file_extension`, `get_file_size_mb`, `is_file_readable`
   - 다른 모듈들의 중복 코드 제거를 위한 중앙 저장소 역할 강화