## RAG 기반 LLM Agentic 자동 통계검정 시스템: 전체 설계도 및 구현 의도

### 1. 프로젝트 비전 및 핵심 아키텍처

#### 비전
본 프로젝트는 사용자가 자연어로 던진 질문 하나만으로, AI Agent가 데이터의 특성을 파악하고, 통계학적 절차에 따라 스스로 최적의 분석을 수행하며, 최종적으로 비즈니스 관점의 인사이트가 담긴 보고서를 생성하는 **완전 자율형 통계 분석 시스템**을 구축하는 것을 목표로 한다.

#### 핵심 아키텍처: `Orchestrator-Engine` 모델
이 시스템의 아키텍처는 **'지휘자(Orchestrator)'**와 **'엔진(Engine)'**이라는 두 가지 핵심 개념으로 분리된다.

* **지휘자 (`core/`)**: 전체 분석의 흐름(파이프라인)을 관리하고, 각 단계에서 어떤 작업이 필요한지를 결정한다. 하지만 실제 복잡한 연산이나 추론은 직접 수행하지 않는다.
* **엔진 (`services/`)**: 지휘자의 요청을 받아 실제적인 작업(LLM 추론, 통계 계산, RAG 검색 등)을 수행하는 강력한 전문 서비스 모음이다.

이러한 분리 구조는 파이프라인을 단순하게 유지하면서도, 각 서비스의 전문성과 재사용성을 극대화하여 유연하고 확장 가능한 시스템을 만든다.

### 2. 최종 5단계 워크플로우

사용자 요청부터 최종 보고서 생성까지의 과정은 다음과 같은 5개의 논리적 단계로 구성된다.

1.  **데이터 선택 (`DataSelectionStep`)**: 사용자가 분석할 데이터를 선택하고 시스템이 이를 로드한다.
2.  **요청 분석 (`UserRequestStep`)**: 사용자의 자연어 요청을 LLM이 분석하여 구조화된 분석 목표(예: 변수, 분석 유형)를 정의한다.
3.  **자율 분석 (`AutonomousAnalysisStep`)**: **(시스템의 심장)** AI Agent가 구조화된 목표에 따라 **[계획 수립 → 가정 검토 → 동적 결정 → 통계 실행 → 사후 검토]**의 전 과정을 자율적으로 수행한다.
4.  **시각화 (`VisualizationStep`)**: 분석 결과를 가장 효과적으로 보여줄 수 있는 시각 자료를 AI Agent가 추천하고 생성한다.
5.  **보고 (`ReportingStep`)**: 모든 수치적, 시각적 결과를 종합하여 인간이 이해하기 쉬운 최종 보고서를 생성한다.

---

### 3. 전체 디렉토리 구조

```
text_to_statistical_test/
├── main.py                      # 애플리케이션 진입점
├── pyproject.toml               # 프로젝트 의존성 및 메타데이터
├── .env                         # 환경 변수 (API 키 등)
├── Dockerfile                   # Docker 이미지 빌드 설정
│
├── 📁 input_data/
│   └── 📁 data_files/              # 원본 데이터 파일 (CSV, Excel 등)
│
├── 📁 output_data/
│   ├── 📁 reports/                  # 최종 생성된 보고서 (HTML, PDF 등)
│   ├── 📁 visualizations/          # 생성된 시각화 이미지 파일
│   └── 📁 vector_store/            # RAG를 위한 벡터 DB 저장소
│
├── 📁 logs/                        # 애플리케이션 로그
│   └── app.log
│
├── 📁 config/                      # 애플리케이션 설정
│   └── settings.py
│
├── 📁 core/                        #  지휘자(Orchestrator) 계층: 워크플로우, Agent 로직
│   ├── 📁 pipeline/                # 5단계 워크플로우 각 단계 모듈
│   │   ├── base_pipeline_step.py
│   │   ├── data_selection_step.py
│   │   ├── user_request_step.py
│   │   ├── autonomous_analysis_step.py
│   │   ├── visualization_step.py
│   │   └── reporting_step.py
│   ├── 📁 agent/                   # Agent의 의사결정 및 도구 사용 로직
│   │   └── ...
│   ├── 📁 reporting/               # 보고서 파일 생성 및 포맷팅
│   │   └── report_builder.py
│   └── 📁 workflow/                # 파이프라인 전체 실행 관리
│       └── orchestrator.py
│
├── 📂 services/                    # 엔진(Engine) 계층: 핵심 기능 수행
│   ├── 📁 llm/                     # LLM 관련 모든 기능
│   │   ├── llm_client.py
│   │   ├── llm_response_parser.py
│   │   ├── prompt_engine.py
│   │   └── llm_service.py         # Facade: LLM 기능 통합 제공
│   ├── 📁 statistics/              # 통계 관련 모든 기능
│   │   ├── assumption_checks.py
│   │   ├── parametric_tests.py
│   │   ├── nonparametric_tests.py
│   │   ├── regression_analysis.py
│   │   ├── categorical_analysis.py
│   │   ├── posthoc_analysis.py
│   │   ├── effect_size.py
│   │   └── stats_service.py       # Facade: 통계 기능 통합 제공
│   ├── 📁 rag/                     # RAG 관련 모든 기능
│   │   ├── vector_store.py
│   │   ├── knowledge_store.py
│   │   ├── retriever.py
│   │   ├── context_builder.py
│   │   └── rag_service.py         # Facade: RAG 기능 통합 제공
│   └── 📁 visualization/           # 시각화 기능
│       └── viz_service.py
│
├── 📂 utils/                       # 범용 유틸리티 계층
│   ├── data_loader.py
│   ├── data_utils.py
│   ├── helpers.py
│   ├── input_validator.py
│   └── helpers.py
│
└── 📁 resources/                   # 지식(Knowledge) 계층
    └── 📁 knowledge_base/          # RAG가 학습하고 참조하는 지식 데이터
        ├── 📁 business_domains/    # 도메인별 용어, KPI, 스키마 정의 (JSON, MD)
        ├── 📁 statistical_concepts/ # 통계 이론, 가정, 해석 방법 (MD)
        └── 📁 code_templates/      # 분석 코드 생성 시 참조할 템플릿 (PY)
```

---

### 4. 주요 디렉토리별 상세 설명 및 구현 의도

#### `core/pipeline/` - 워크플로우의 단계별 정의
* **구현 의도**: 각 `_step.py` 파일은 **자신이 무엇을 해야 하는지만 알고, 어떻게 하는지는 모릅니다.** 예를 들어, `AutonomousAnalysisStep`은 "상세 분석 계획을 수립하고, 가정을 검토한 뒤, 분석을 실행해야 한다"는 **흐름만 알고 있습니다.** 실제 계획 수립은 `LLMService`에, 가정 검토와 분석 실행은 `StatisticsService`에 위임합니다. 이로 인해 파이프라인 코드는 매우 간결하고 읽기 쉬워집니다.

#### `services/` - 시스템의 실제 두뇌와 손발
* **`services/llm/`**: AI Agent의 **'두뇌(추론, 계획, 생성)'** 역할을 합니다.
    * `llm_service.py`: 파이프라인의 요청을 받아, `PromptEngine`으로 질문지를 만들고 `LLMClient`로 API를 호출한 뒤, `LLMResponseParser`로 답변을 정리하여 돌려주는 모든 과정을 책임집니다.
* **`services/statistics/`**: AI Agent의 **'계산기/실행 도구'** 역할을 합니다.
    * `stats_service.py`: `LLMService`가 수립한 계획("독립표본 t-검정 실행해줘")을 받아, `parametric_tests.py` 등 하위 모듈의 실제 함수를 호출하여 통계 계산을 수행하고 결과를 반환합니다.
* **`services/rag/`**: AI Agent의 **'외부 기억/참고서'** 역할을 합니다.
    * `rag_service.py`: `LLMService`가 더 똑똑한 판단을 내릴 필요가 있을 때(예: "회귀분석 코드 어떻게 짜야 하지?"), `resources/knowledge_base`에서 가장 관련 있는 지식(이론, 코드 템플릿)을 찾아 제공합니다.

#### `resources/knowledge_base/` - AI Agent의 지식 원천
* **구현 의도**: 이 디렉토리의 내용을 풍부하게 채울수록 AI Agent의 전문성이 향상됩니다. 특정 산업군의 분석을 자주 수행한다면 `business_domains/`에 해당 산업의 용어와 KPI, DB 스키마를 추가하고, 새로운 통계 기법을 추가하고 싶다면 `statistical_concepts/`와 `code_templates/`에 관련 내용을 추가하는 것만으로 시스템을 손쉽게 확장할 수 있습니다.

### 5. 데이터 및 실행 흐름 요약
1.  **시작 (`main.py`)**: `Orchestrator`가 파이프라인 실행을 시작합니다.
2.  **1단계 (데이터)**: `DataSelectionStep`이 `DataLoader`를 통해 데이터를 로드합니다.
3.  **2단계 (요청)**: `UserRequestStep`이 `LLMService`를 호출하여 사용자의 말을 '구조화된 분석 목표(JSON)'로 번역합니다.
4.  **3단계 (분석)**:
    * `AutonomousAnalysisStep`이 `LLMService`를 호출하여 '구조화된 목표'를 '상세 실행 계획(JSON)'으로 발전시킵니다.
    * 이후 `StatisticsService`를 호출하여 '상세 실행 계획'에 명시된 가정 검토, 본 분석, 사후 검정을 순차적으로 수행하고, 그 결과를 받습니다.
5.  **4단계 (시각화)**: `VisualizationStep`이 분석 결과를 `LLMService`에 보내 "어떤 그래프가 좋을까?"라고 묻고, 추천받은 내용을 `VisualizationService`에 전달하여 이미지 파일을 생성합니다.
6.  **5단계 (보고)**: `ReportingStep`이 3단계의 통계 결과와 4단계의 이미지 파일 경로를 `LLMService`에 전달하여 최종 보고서의 내용을 생성하고, `ReportBuilder`를 통해 파일로 저장합니다.
7.  **종료**: 최종 보고서의 경로가 사용자에게 반환됩니다.