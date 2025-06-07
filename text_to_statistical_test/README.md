# Text-to-Statistical-Test(TTST) 📊

자연어 질문을 통해 복잡한 통계 분석을 자율적으로 수행하고, 이해하기 쉬운 보고서를 생성하는 AI 에이전트 시스템입니다. 더 이상 통계 지식이나 코딩 능력 때문에 데이터 분석에 어려움을 겪지 마세요. 당신의 질문이 곧 분석 결과가 됩니다.

---

## 🚀 주요 목적 및 특징

이 시스템의 핵심 목표는 **데이터 분석의 민주화**입니다. 통계 전문가가 아니더라도 누구나 자신의 데이터를 기반으로 깊이 있는 인사이트를 얻을 수 있도록 돕습니다.

- **자연어 기반 분석**: "A와 B 그룹 간의 평균 차이가 있나요?"와 같은 간단한 질문만으로 분석을 시작할 수 있습니다.
- **자율 분석 에이전트**: AI 에이전트가 데이터의 특성과 질문의 의도를 파악하여, 최적의 통계 기법을 스스로 선택하고 분석 계획을 수립합니다.
- **자동 보고서 생성**: 분석 과정과 결과를 종합하여, 주요 발견 사항과 결론이 담긴 명확한 Markdown 보고서를 생성합니다.
- **유연하고 확장 가능한 아키텍처**: 새로운 통계 분석 기능을 쉽게 추가할 수 있는 `Orchestrator-Engine` 모델을 채택하여 지속적인 발전이 가능합니다.

## 🛠️ 설치 방법

이 시스템은 Poetry를 사용하여 프로젝트 의존성을 관리합니다. 다음 단계를 따라 설치를 진행해 주세요.

1.  **GitHub 리포지토리 복제(Clone)**:
    ```bash
    git clone https://github.com/wooshikwon/text-to-statistical-test.git
    cd text-to-statistical-test
    ```

2.  **Poetry 설치**:
    아직 Poetry가 없다면, 공식 가이드를 따라 설치합니다.
    ```bash
    # macOS / Linux / WSL
    curl -sSL https://install.python-poetry.org | python3 -
    ```

3.  **프로젝트 의존성 설치**:
    프로젝트 루트 디렉토리에서 다음 명령을 실행하여 필요한 모든 패키지를 가상 환경에 설치합니다.
    ```bash
    poetry install
    ```

4.  **환경 변수 설정**:
    OpenAI API를 사용하기 위해, 프로젝트 루트에 `.env` 파일을 생성하고 다음과 같이 API 키를 입력해야 합니다.
    ```
    # .env 파일
    OPENAI_API_KEY="sk-..."
    ```

## 📈 지원하는 통계 검정 목록

현재 시스템은 다음과 같은 광범위한 통계 분석을 지원합니다.

- **그룹 간 평균 비교**:
    - 독립 표본 t-검정 (Independent Samples t-test)
    - 대응 표본 t-검정 (Paired Samples t-test)
    - 일원 분산 분석 (One-way ANOVA)
- **관계 분석**:
    - 선형 회귀 분석 (Linear Regression)
    - 카이제곱 독립성 검정 (Chi-Square Independence Test)
- **비율 분석 (A/B 테스트 등)**:
    - 단일 표본 비율 검정 (One-Proportion Test)
    - 두 표본 비율 검정 (Two-Proportion Test)
- **분포 비교**:
    - 카이제곱 적합도 검정 (Chi-Square Goodness-of-Fit Test)
- **비모수 검정**:
    - Mann-Whitney U 검정
    - Kruskal-Wallis H 검정
    - Wilcoxon 부호-순위 검정

## 📖 사용 방법

시스템 사용법은 매우 간단합니다. 터미널에서 `main.py`를 실행하며, 분석할 **파일 경로**와 **분석 요청**을 인자로 전달하면 됩니다.

### 1. 데이터 파일 준비

-   분석할 데이터를 `.csv` 형식의 파일로 준비합니다.
-   파일의 첫 번째 행은 반드시 각 열의 의미를 나타내는 **헤더(Header)**여야 합니다.
-   준비된 파일을 프로젝트 내의 `input_data/data_files/` 디렉토리 안에 위치시킵니다. (다른 경로도 무방하지만, 이 경로를 권장합니다.)

**예시: `anova_customer_satisfaction.csv`**
```csv
Location,Satisfaction
Downtown,85
Suburban,78
Mall,92
Online,75
...
```

### 2. 명령어 실행

터미널에서 `poetry run python main.py` 명령어와 함께 `--file`과 `--request` 옵션을 사용합니다.

-   `--file`: 분석할 데이터 파일의 경로.
-   `--request`: 자연어로 작성된 분석 요청.

**사용 예시:**

-   **ANOVA 분석**:
    ```bash
    poetry run python main.py \
      --file "input_data/data_files/anova_customer_satisfaction.csv" \
      --request "매장 위치(Location)별 고객 만족도(Satisfaction)에 유의미한 차이가 있는지 분산 분석을 통해 알려줘."
    ```

-   **선형 회귀 분석**:
    ```bash
    poetry run python main.py \
      --file "input_data/data_files/regression_housing_prices.csv" \
      --request "집의 크기(SquareFootage)가 주택 가격(Price)에 어떤 영향을 미치는지 회귀 분석으로 설명해줘."
    ```

-   **두 표본 비율 검정 (A/B 테스트)**:
    ```bash
    poetry run python main.py \
      --file "input_data/data_files/ab_test_click_data.csv" \
      --request "A 디자인과 B 디자인의 클릭률(clicked)에 차이가 있는지 분석해줘."
    ```

분석이 완료되면, 결과 보고서가 터미널에 직접 출력되고, `output_data/reports/` 디렉토리에 Markdown 파일로도 저장됩니다.

## ⚙️ 작동 원리 (How it Works)

이 시스템은 **지휘자(Orchestrator)**와 **엔진(Engine)**이라는 두 가지 핵심 개념을 기반으로 설계되었습니다. 이 아키텍처는 전체 분석 과정을 체계적으로 관리하며, 복잡한 작업을 전문화된 모듈에 위임하여 시스템의 유연성과 확장성을 극대화합니다.

1.  **Orchestrator (`core/`)**: 전체 분석 파이프라인의 흐름을 지휘합니다. 사용자의 요청이 들어오면, 정해진 순서에 따라 각 단계를 실행하고 데이터가 올바르게 전달되도록 관리합니다. 실제 통계 계산이나 AI 추론 같은 복잡한 작업은 직접 수행하지 않고, 각 분야의 전문 '엔진'에게 위임합니다.

2.  **Engines (`services/`)**: 특정 작업을 수행하는 강력한 전문 서비스 모음입니다.
    -   **`LLMService`**: 자연어 요청을 이해하고, 분석 계획을 수립하며, 최종 결과를 해석하는 두뇌 역할을 합니다.
    -   **`StatisticsService`**: t-검정, ANOVA, 회귀 분석 등 실제 통계 연산을 담당합니다.
    -   **`ReportService`**: 모든 분석 결과를 종합하여 사람이 이해하기 쉬운 Markdown 보고서를 생성합니다.

이 모든 과정은 다음과 같은 **3단계 워크플로우**를 통해 자동으로 진행됩니다.

1.  **데이터 로드**: 사용자가 지정한 데이터 파일을 시스템으로 불러옵니다.
2.  **자율 분석**: AI 에이전트가 데이터와 요청을 기반으로 **[계획 수립 → 통계 분석 실행 → 결과 종합]**의 과정을 거쳐 최적의 분석을 수행합니다.
3.  **보고서 생성**: 분석 결과를 바탕으로 최종 보고서를 생성하여 사용자에게 제공합니다.

더 상세한 아키텍처 설계가 궁금하시다면 [ARCHITECTURE.md](ARCHITECTURE.md) 문서를 참고해 주세요.