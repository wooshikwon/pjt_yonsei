LLM Agent 통계 분석 시스템 구축 계획서

1. 프로젝트 개요
프로젝트명: text_to_statistical_test
미션: 사용자가 제공한 테이블 형식의 데이터 파일(csv, parquet, xlsx 등)과 자연어 분석 요청을 입력받아, RAG(검색 증강 생성)를 통해 데이터의 비즈니스 맥락을 파악하고, 통계적으로 유의미한 분석 절차를 자율적으로 계획하고 수행하여 최종 분석 보고서를 생성하는 LLM 에이전트를 구축합니다.
핵심 가치: 복잡한 통계 지식이나 코딩 능력 없이도 누구나 자연어 질문만으로 데이터 기반의 전문적인 통계 분석 및 의사결정을 내릴 수 있도록 지원합니다.
중요 사항: RAG를 .env에서 끄거나 켤 수 있게 만들어야 합니다.

2. 목표 시스템 명세
입력 (Input):

데이터 파일: csv, parquet, xlsx 등 pandas로 처리가능한 테이블 형식의 파일
자연어 요청: 분석의 목적과 대상이 담긴 한국어 자연어 문장 (e.g., "A와 B 제품 간의 고객 만족도에 차이가 있는지 비교 분석해줘")

핵심 프로세스 (Process):

맥락 이해 (Contextual Understanding): RAG를 통해 내부 지식 베이스(비즈니스 용어 정의, 데이터 사전 등)를 검색하여, 사용자의 자연어 요청(e.g., '고객 만족도')과 데이터 파일 내의 실제 컬럼(e.g., satisfaction_score)을 연결하고 분석의도를 명확히 합니다.
자율적 분석 계획 수립 (Autonomous Planning): 데이터의 구조와 통계적 특성을 탐색한 후, 최종 목표에 도달하기 위한 전체 통계 분석 절차를 단계별로 상세하게 수립합니다. 이 계획에는 데이터 전처리, 사전 검정(정규성, 등분산성), 본 검정(t-test, 비율 검정, One-way, Two-way ANOVA, linear, logistic 회귀분석 등), 사후 검정(Cohen's d, Tukey's HSD 등)이 모두 포함됩니다.
코드 생성 및 실행 (Code Generation & Execution): 수립된 계획에 따라 각 단계에 필요한 Python 코드를 생성하고, 안전한 환경에서 실행하여 결과를 확인합니다.
자가 수정 (Self-Correction): 코드 실행 중 오류가 발생할 경우, 오류의 원인을 파악하고 스스로 코드를 수정하여 과업을 재시도합니다.
중요 사항: RAG를 .env에서 끄거나 켤 수 있게 만들어야 합니다.

출력 (Output):

터미널 출력: 분석 보고서 전문을 터미널에 실시간으로 출력합니다.
보고서 파일 저장: 동일한 내용의 분석 보고서를 마크다운(.md) 파일 형태로 지정된 경로에 저장합니다.
보고서 구성:
주요 발견 사항 (Key Findings): 분석 결과 중 가장 중요하고 핵심적인 내용을 요약하여 전달합니다. (e.g., "A 제품의 고객 만족도 점수는 B 제품보다 통계적으로 유의미하게 높았습니다 (p < 0.05).")
결론 및 권장 사항 (Conclusion & Recommendations): 분석 결과를 바탕으로 내릴 수 있는 비즈니스적 결론과 구체적인 후속 조치 사항을 제안합니다. (e.g., "B 제품의 만족도 개선을 위한 사용자 경험(UX) 분석을 긴급히 진행할 것을 권장합니다.")
통계 검정 상세 결과 (Detailed Results): 수행된 모든 통계 검정의 상세 수치(검정 통계량, p-value, 효과 크기 등)와 시각화 자료(필요시)를 투명하게 제공합니다.

3. 기술 스택 및 프로젝트 아키텍처
언어: Python 3.10+
패키지 및 가상환경: Poetry
LLM: OpenAI GPT-4o (에이전트의 추론 및 생성용)
RAG Framework:
Retriever: LlamaIndex (효율적인 인덱싱 및 검색 파이프라인 구축에 용이)
Embedding Model:
  - 모델: `jhgan/ko-sroberta-multitask` (Sentence-Transformers 기반)
  - 특징: 한국어에 특화된 로컬 임베딩 모델로, API 키가 필요 없어 안정적이고 비용 효율적임.
Vector Store: FAISS (빠른 속도와 로컬 환경 구동에 최적화)
핵심 라이브러리: openai, pandas, scipy, statsmodels, scikit-learn, openpyxl, pyarrow, typer (CLI), python-dotenv, llama-index, faiss-cpu, sentence-transformers
배포: Docker

📂 디렉토리 구조
text_to_statistical_test/
├── .dockerignore
├── .env                  # OpenAI API 키, RAG 동작 제어 등 민감 정보 및 설정 관리
├── env.example           # .env 파일의 템플릿
├── Dockerfile            # Docker 이미지 빌드 설정
├── pyproject.toml        # Poetry 의존성 및 프로젝트 설정
├── README.md
├── input_data/
│   └── data_files/       #  분석할 원본 데이터 파일 저장
│       └── customer_data.csv
├── output_data/
│   └── reports/          # 생성된 분석 보고서 저장
│       └── report-20250618-1530.md
├── resources/
│   ├── knowledge_base/   # RAG 검색 대상 문서 (e.g., business_glossary.md)
│   └── rag_index/        # 생성된 FAISS 벡터 인덱스 파일 저장
└── src/
    ├── __init__.py
    ├── main.py             # CLI 시작점, Orchestrator 실행
    ├── agent.py            # LLM Agent (프롬프트 관리, LLM API 호출)
    ├── components/
    │   ├── __init__.py
    │   ├── context.py        # 시스템의 모든 상태를 관리하는 State Manager 클래스
    │   ├── rag_retriever.py  # RAG 검색 및 임베딩 처리 모듈
    │   └── code_executor.py  # 생성된 Python 코드를 안전하게 실행하는 모듈
    └── prompts/
        └── system_prompts.py # 역할/작업별 시스템 프롬프트 템플릿 관리

4. 핵심 컴포넌트 설계
1) Orchestrator (main.py)
역할: 전체 분석 파이프라인을 총괄 지휘하는 컨트롤 타워.
기능:
Typer를 사용하여 터미널 명령어 인자(--file, --request)를 파싱.
Context 객체를 생성하고, 각 모듈(RAG, Agent, Executor)을 정해진 순서에 맞게 호출하며 Context를 지속적으로 업데이트.
Agent가 수립한 통계 분석 계획에 따라 실행 루프를 관리하고, 각 단계를 순차적으로 요청.
최종 보고서를 받아 출력 및 저장하며 프로세스를 종료.

2) State Manager (context.py의 Context 클래스)
역할: 에이전트의 "작업 기억 공간(Working Memory)". 시스템의 모든 상태와 데이터를 구조화하여 저장.
포함 정보 예시:
user_input: { 'file_path': 'input_data/data_files/customer_data.csv', 'request': 'A와 B 제품 간의 고객 만족도에 차이가 있는지 비교 분석해줘' }
rag_results: ['고객 만족도는 satisfaction_score 컬럼을 의미함. 점수는 1-5점 척도.', '제품 구분은 product_name 컬럼을 따름.']
data_info: { 'schema': {'product_name': 'object', 'satisfaction_score': 'int64'}, 'null_values': {'satisfaction_score': 0}, 'sample_data': df.head().to_string() }
analysis_plan: ['1. product_name 컬럼에서 "A", "B" 데이터 필터링', '2. A 제품 만족도 점수의 정규성 검정 (Shapiro-Wilk)', ...]
conversation_history: [{'role': 'system', 'content': '...'}, {'role': 'assistant', 'code': '...'}, {'role': 'user', 'result': '...'}, ...]
final_report: "# 최종 분석 보고서\n## 1. 주요 발견 사항\n..."

3) RAG Retriever (rag_retriever.py)
역할: 사용자의 요청과 데이터를 비즈니스 맥락과 연결하는 지식 탐색가.
기능:
임베딩 및 인덱스 관리: `.env` 파일의 `REBUILD_VECTOR_STORE=True` 설정 시, 기존 `rag_index`를 삭제하고 `knowledge_base`의 문서를 다시 임베딩하여 FAISS 인덱스를 새로 생성. 평상시(`False`)에는 기존에 생성된 인덱스를 로드하여 사용.
정보 검색: `.env` 파일의 `USE_RAG=True`일 경우에만 활성화. 사용자 요청이 들어오면, 요청과 데이터 컬럼명 등을 조합하여 쿼리를 생성. `rag_index`의 FAISS 인덱스에서 가장 관련성 높은 문서 내용을 검색하여 Context에 추가.
예시: request "고객 만족도"와 데이터 컬럼 satisfaction_score를 키워드로 knowledge_base를 검색하여 "고객 만족도는 1-5점 척도"라는 정보를 찾아냄.

4) LLM Agent (agent.py)
역할: OpenAI API와 통신하며 주어진 과업을 수행하는 시스템의 두뇌. 명확히 정의된 Task 단위로 동작.
주요 Tasks:
Task 1: 데이터 탐색 코드 생성: Context를 입력받아 데이터의 기본 정보(스키마, 통계 요약 등)를 파악하기 위한 df.info(), df.describe() 등의 코드를 JSON 형식으로 반환. {"thought": "데이터의 기본 구조와 타입을 파악해야겠다.", "code": "print(df.info())"}
Task 2: 통계 분석 계획 수립: 데이터 탐색 결과와 RAG 컨텍스트를 종합하여, 전체 통계 분석 절차를 구체적인 단계별 목록으로 작성하여 반환.
Task 3: 단계별 코드 생성: Orchestrator가 요청하는 현재 계획 단계에 해당하는 Python 코드를 생성하여 반환.
Task 4: 코드 자가 수정: Code Executor가 반환한 오류 메시지를 Context를 통해 받고, 문제를 해결하기 위한 수정된 코드를 생성하여 반환.
Task 5: 최종 보고서 작성: 모든 분석이 완료된 최종 Context를 받아, 지정된 형식(주요 발견, 결론/권장, 상세 결과)에 맞춰 보고서를 Markdown으로 작성.

5) Secure Code Executor (code_executor.py)
역할: Agent가 생성한 코드를 실행하는 안전한 격리 실행기.
기능:
exec() 또는 subprocess를 사용하여 코드 실행 환경을 분리.
핵심 기능: try-except 구문을 통해 코드 실행을 감싸고, 성공 시 stdout(출력 결과)을, 실패 시 stderr(오류 메시지)를 정확히 캡처하여 문자열 형태로 Orchestrator에 반환.
예시: Agent가 print(df['score']) 코드를 생성했으나 실제 컬럼명이 satisfaction_score일 경우, KeyError: 'score'라는 오류 메시지를 정확히 포착하여 반환.


5. 상세 실행 파이프라인 및 시나리오 예시
시나리오: 마케팅 분석가가 A/B 테스트 결과를 분석하고자 함.

[Step 0] 사용자 실행 및 환경 설정
사용자는 프로젝트 루트에 `.env` 파일을 생성하고 아래와 같이 설정을 구성할 수 있습니다.
- `USE_RAG=True` / `False` : RAG 사용 여부 제어
- `REBUILD_VECTOR_STORE=True` / `False` : RAG 인덱스 재생성 여부 제어
- `OPENAI_API_KEY="sk-..."` : OpenAI API 키 설정

터미널에 다음 명령어를 입력합니다.

Bash

poetry run python src/main.py --file "customer_data.csv" --request "A와 B 제품 간의 고객 만족도에 차이가 있는지 비교 분석해줘"
[Step 1] 초기화 및 컨텍스트 강화

Orchestrator는 `.env` 파일을 읽어 RAG 사용 여부(`USE_RAG`)를 확인합니다.
`USE_RAG`가 `True`일 경우, RAG Retriever를 초기화하고 `REBUILD_VECTOR_STORE` 값을 확인하여 인덱스를 로드하거나 재생성합니다.
RAG Retriever는 request의 '고객 만족도'와 customer_data.csv의 컬럼 정보를 바탕으로 `knowledge_base`를 검색합니다.
결과 (예시): "비즈니스 용어집.md"에서 다음 내용을 찾아 Context에 추가합니다.
"고객 만족도: satisfaction_score 컬럼을 의미하며, 1점(매우 불만족)부터 5점(매우 만족)까지의 정수형 데이터. 제품 구분은 product_name 컬럼 ('A', 'B')을 따름."
`USE_RAG`가 `False`일 경우, 이 단계를 건너뜁니다.

[Step 2] 데이터 탐색 (Agent - Task 1)

Orchestrator는 Agent에게 "데이터 탐색 코드 생성"을 요청합니다.
Agent는 {"code": "print(df.info()); print(df.describe())"} 와 같은 코드를 반환합니다.
Code Executor가 코드를 실행하고, 결과(컬럼 타입, 결측치 유무, 데이터 수 등)를 Context에 추가합니다.

[Step 3] 통계 분석 계획 수립 (Agent - Task 2)

Orchestrator는 RAG 결과와 데이터 탐색 결과가 포함된 Context를 Agent에게 전달하며 "통계 분석 계획 수립"을 요청합니다.
Agent는 두 독립 표본의 평균 비교 시나리오임을 인지하고, 다음과 같은 구체적인 계획 리스트를 생성하여 Context에 저장합니다.

Python

analysis_plan = [
    '1. 데이터에서 product_name이 "A" 또는 "B"인 행만 필터링합니다.',
    '2. A 제품 그룹의 satisfaction_score에 대한 정규성 검정(Shapiro-Wilk)을 수행합니다.',
    '3. B 제품 그룹의 satisfaction_score에 대한 정규성 검정(Shapiro-Wilk)을 수행합니다.',
    '4. 두 그룹의 satisfaction_score에 대한 등분산성 검정(Levene)을 수행합니다.',
    '5. 정규성 및 등분산성 검정 결과에 따라, 독립표본 t-검정(Independent T-test) 또는 Welch의 t-검정을 실행하여 평균 차이를 비교합니다.',
    '6. 통계적으로 유의미한 차이가 발견될 경우, 효과 크기(Cohen\'s d)를 계산하여 차이의 정도를 측정합니다.'
]

[Step 4] 계획 기반 실행 루프 (Agent - Task 3)

Orchestrator는 analysis_plan의 첫 번째 단계('1. ...필터링')를 Agent에게 전달하며 "단계별 코드 생성"을 요청합니다.
Agent는 df_filtered = df[df['product_name'].isin(['A', 'B'])]와 같은 코드를 생성합니다.
Executor는 코드를 실행하고, 성공적으로 실행되었음을 알립니다. 데이터프레임은 내부적으로 업데이트 됩니다.
Orchestrator는 analysis_plan의 마지막 단계가 끝날 때까지 이 과정을 반복합니다.

[Step 5] 오류 처리 및 자가 수정 시나리오 (Agent - Task 4)

상황: analysis_plan의 2단계 코드를 생성할 때 Agent가 실수로 존재하지 않는 컬럼명을 사용. scipy.stats.shapiro(df_filtered['score_satisfaction'])
실행: Code Executor는 코드를 실행하다 KeyError: 'score_satisfaction' 오류를 만납니다.
피드백: Executor는 이 오류 메시지를 Orchestrator에게 반환합니다.
수정 요청: Orchestrator는 Agent에게 다음과 같은 정보를 포함하여 "코드 자가 수정"을 요청합니다.
"이전 코드 실행 시 KeyError: 'score_satisfaction' 오류가 발생했습니다. 사용 가능한 컬럼은 ['product_name', 'satisfaction_score'] 입니다. 코드를 수정해주세요."

자가 수정: Agent는 오류와 컨텍스트를 이해하고 scipy.stats.shapiro(df_filtered['satisfaction_score']) 라는 수정된 코드를 반환합니다. Orchestrator는 이 수정된 코드로 해당 단계를 재시도합니다.

[Step 6] 최종 보고서 생성 (Agent - Task 5)

모든 계획이 성공적으로 실행되면, Orchestrator는 모든 과정과 결과가 담긴 최종 Context를 Agent에게 전달하며 "최종 보고서 작성"을 요청합니다.
Agent는 Context 내용을 종합하여 아래와 같은 형식의 보고서를 Markdown으로 생성합니다.

[Step 7] 종료

Orchestrator는 Agent가 생성한 보고서를 받아 터미널에 출력하고, output_data/reports/report-YYYYMMDD-HHMM.md 파일로 저장하며 모든 프로세스를 성공적으로 마칩니다.


***  Agent가 '최종 절차를 마무리했다'는 시스템과의 소통을 강건하게 하기 위해 아래와 같은 권장 전략을 따를 것

권장 전략: Orchestrator 주도 상태 관리와 Agent의 명시적 신호 결합
이 방식은 두 가지 요소를 결합하여 서로를 보완하고 검증합니다.

Orchestrator의 상태 관리: Orchestrator는 analysis_plan 리스트 전체를 알고 있으며, 현재 몇 번째 단계를 실행 중인지 내부적으로 추적합니다.
Agent의 명시적 신호: Agent는 코드 생성 시, 해당 코드가 계획의 마지막 단계에 해당하는지를 명시적으로 표현하여 전달합니다.
상세 실행 흐름 (Handshake Protocol)
아래는 이 전략을 구현하는 구체적인 통신 절차입니다.

1. Orchestrator: "마지막 단계 요청" 인지 및 힌트 제공

Orchestrator는 자신의 실행 루프 내에서 현재 실행할 단계가 계획의 마지막인지 스스로 인지합니다.

Python

# Orchestrator 로직 (main.py 내)

analysis_plan = context.get('analysis_plan')
for i, step_description in enumerate(analysis_plan):
    is_final_step_in_plan = (i == len(analysis_plan) - 1)

    # Agent에게 현재 단계가 마지막임을 '힌트'로 알려줄 수 있음 (선택 사항이지만 권장)
    # 이는 Agent가 더 정확한 응답을 하도록 유도
    response = agent.generate_code_for_step(
        context=context,
        step_description=step_description,
        is_final_hint=is_final_step_in_plan
    )
    
    # ... (이후 응답 처리)
2. Agent: "마지막 단계"임을 명시한 구조화된 응답

Agent는 요청을 받고 코드를 생성할 때, 이 단계가 마지막임을 인지하고 응답 JSON에 명시적인 status 필드를 포함하여 반환합니다.

JSON

/* Agent 응답 (JSON 예시) */
{
  "thought": "이것이 분석 계획의 마지막 단계인 효과 크기 계산입니다. 이 작업이 성공적으로 끝나면 모든 분석 절차가 완료됩니다.",
  "code": "from math import sqrt\n\n# ... (Cohen's d 계산 코드) ...\nprint(f'Cohen\\'s d: {cohen_d}')",
  "status": "FINAL_ANALYTICAL_STEP"  // <--- 이것이 핵심 신호!
}
status 필드는 'INTERMEDIATE_STEP'(중간 단계) 또는 'FINAL_ANALYTICAL_STEP'(최종 분석 단계)과 같은 약속된 값을 가질 수 있습니다.
3. Orchestrator: 이중 검증 및 상태 전환

Orchestrator는 Agent의 응답을 받고, 자신의 상태와 Agent의 신호를 이중으로 검증합니다.

Python

# Orchestrator 로직 (main.py 내) - 계속

# 1. Agent가 보낸 상태 신호 확인
agent_status = response.get('status')

# 2. 이중 검증 (Cross-validation)
if is_final_step_in_plan and agent_status == "FINAL_ANALYTICAL_STEP":
    # Orchestrator와 Agent 모두 마지막 단계임을 동의함 -> 강건한 상태
    print("INFO: Final analytical step confirmed. Executing...")
    
    # 마지막 단계 코드 실행
    execution_result, success = code_executor.run(response['code'])

    if success:
        print("INFO: Final step executed successfully. Transitioning to reporting phase.")
        # 모든 분석이 성공적으로 끝났으므로, 루프를 종료하고 보고서 생성 단계로 넘어감
        context.add_to_history(response['code'], execution_result)
        # break # 루프를 빠져나감
    else:
        # 마지막 단계 실행 실패! 보고서 생성으로 넘어가면 안 됨.
        # 자가 수정 루프를 시작
        print("ERROR: Final step failed. Initiating self-correction...")
        # ... 자가 수정 로직 호출 ...
else:
    # 상태 불일치 에러: Orchestrator는 마지막이라 생각했는데 Agent는 아니라고 하거나, 그 반대의 경우
    # 이것은 심각한 로직 오류일 수 있으므로, 예외 처리
    raise SystemError("State mismatch between Orchestrator and Agent about the final step.")

4. 최종 상태 전환: 보고서 생성 요청

실행 루프가 성공적으로 완료된 후에야 (break 되거나 루프가 자연스럽게 끝난 후), Orchestrator는 비로소 Agent에게 Task 5: 최종 보고서 작성을 요청합니다.

Python

# Orchestrator 로직 (main.py 내) - 루프 종료 후

# 모든 계획이 성공적으로 실행되었음이 보장된 상태
final_report = agent.generate_final_report(context)
print(final_report)
save_report(final_report)