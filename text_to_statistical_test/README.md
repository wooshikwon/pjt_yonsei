text_to_statistical_test/
├── main_runner.py                 # 애플리케이션 실행 스크립트 (CLI 인터페이스 등)
│
├── core/
│   ├── __init__.py
│   ├── agent.py                   # LLMAgent: 워크플로우 오케스트레이션, 상태, 컨텍스트 관리
│   ├── workflow_manager.py        # WorkflowManager: JSON 워크플로우 로드, 파싱, 노드 정보 제공
│   ├── decision_engine.py         # DecisionEngine: LLM 응답, 사용자 입력 기반 조건 판단 및 다음 노드 결정
│   └── context_manager.py         # ContextManager: 대화/분석 이력 관리, 요약, 토큰 최적화 (신규)
│
├── llm_services/
│   ├── __init__.py
│   ├── llm_client.py              # LLMClient: LLM API 연동 (OpenAI, Gemini 등)
│   ├── prompt_crafter.py          # PromptCrafter: 템플릿 기반 상황별 프롬프트 생성 전문
│   └── prompts/                   # (신규 디렉토리) 프롬프트 템플릿 저장
│       ├── common/
│       │   └── output_format_instructions.md
│       ├── stage_1_user_understanding/
│       │   ├── 1_1_analyze_user_request.md
│       │   └── 1_2_confirm_analysis_goal.j2 # Jinja2 템플릿 예시
│       └── ... (기타 단계별/목적별 프롬프트 템플릿)
│
├── data_processing/
│   ├── __init__.py
│   └── data_loader.py             # DataLoader: Tableau 등 데이터 로드 및 기본 정보 추출
│
├── rag_system/
│   ├── __init__.py
│   ├── code_retriever.py          # CodeRetriever: 코드 스니펫 검색 (Vector DB 연동 등)
│   └── code_indexer.py            # CodeIndexer: (셋업 시 사용) 코드 스니펫 임베딩 및 인덱싱
│
├── code_execution/
│   ├── __init__.py
│   └── safe_code_executor.py      # SafeCodeExecutor: RAG로 검색된 통계 코드 안전하게 실행
│
├── reporting/
│   ├── __init__.py
│   └── report_generator.py        # ReportGenerator: 분석 결과 및 과정을 종합하여 보고서 생성
│
├── resources/
│   ├── workflow_graph.json        # 분기 그래프 (상태 및 전환 로직)
│   └── code_snippets/             # 생성된 통계 검정 예시 코드 저장 폴더 (RAG 대상)
│       ├── t_test/
│       │   └── independent_t_test_example_1.py
│       └── ...
│
├── input_data/
│   └── your_data.hyper            # 입력 Tableau 데이터
│
├── output_results/
│   └── analysis_report.md         # 최종 분석 보고서 (Markdown 등)
│
├── config/
│   ├── __init__.py  # 오타 수정 (init.py -> __init__.py)
│   └── settings.py                # 환경변수 로드, 경로, 모델명, 로깅 설정 등
│
├── .env                           # (Git에서 제외됨) 실제 API 키 및 민감 정보 저장
├── .env.example                   # .env 파일 형식 예시
└── .gitignore                     # Git 버전 관리 제외 목록


### 1. `main_runner.py`

- **기능**: 애플리케이션의 최상위 진입점. 의존성 설정, Agent 초기화 및 실행.
- **역할**: 시스템 부트스트랩 및 실행 흐름 제어.
- **주요 함수/클래스**:
    - `setup_dependencies() -> dict`: 각 서비스(LLMClient, DataLoader, ContextManager 등)의 인스턴스를 생성하고, 설정값(`config.settings`)을 기반으로 초기화하여 딕셔너리 형태로 반환.
    - `run_agent_workflow(dependencies: dict, input_data_path: str)`: `LLMAgent` 인스턴스를 생성(주입된 의존성 사용)하고, `agent.run(input_data_path)`를 호출하여 분석 워크플로우 시작.
    - `main()`: CLI 인자 파싱 (예: `argparse` 사용), `setup_dependencies` 호출, `run_agent_workflow` 호출.

### 2. `core/agent.py`

- **기능**: 전체 통계 분석 워크플로우를 오케스트레이션. 현재 상태, 데이터, 분석 관련 중요 정보(변수, 가설 등) 및 대화/작업 이력을 관리.
- **역할**: 중앙 컨트롤 타워. 상태 기계의 실행자.
- **주요 함수/클래스**:
    - `class LLMAgent`:
        - `__init__(self, workflow_mngr: WorkflowManager, decision_eng: DecisionEngine, ctx_mngr: ContextManager, llm_cli: LLMClient, prompt_crftr: PromptCrafter, data_ldr: DataLoader, code_rtrvr: CodeRetriever, code_exec: SafeCodeExecutor, report_gen: ReportGenerator)`: 필요한 모든 서비스/매니저 의존성 주입.
        - `current_node_id: str`: 현재 워크플로우 노드의 ID.
        - `raw_data: pd.DataFrame`: 로드된 원본 데이터.
        - `processed_data: pd.DataFrame`: 전처리/변환된 데이터.
        - `analysis_parameters: dict`: 분석 과정에서 확정된 주요 파라미터 (예: 종속/독립 변수, 선택된 검정 방법, 가설 등).
        - `user_interaction_history: list`: (선택적) 사용자 주요 결정/피드백 기록.
        - `run(self, input_data_path: str) -> str`: 전체 분석 프로세스 시작. 데이터 로드, 초기 노드 설정, 메인 루프 실행. 최종 보고서 경로 반환.
        - `_main_loop(self)`: 현재 노드 처리 -> 다음 노드 결정 -> 상태 전이 반복. 워크플로우 종료 조건 만족 시 루프 종료.
        - `_process_current_node(self)`: 현재 `current_node_id`에 해당하는 노드의 작업을 수행.
            - 노드 타입에 따라 LLM 질의, 사용자 입력 요청(CLI), 데이터 처리, 코드 실행, RAG 검색 등 다양한 액션 분기.
            - LLM 질의 시 `context_manager.get_relevant_history()`를 통해 최적화된 컨텍스트 사용.
            - 모든 상호작용(LLM, 사용자, 시스템 액션)은 `context_manager.add_interaction()`을 통해 기록.
        - `_handle_llm_interaction(self, node_details: dict, current_prompt_context: dict) -> str`: 특정 노드에 대한 LLM 질의 수행. `prompt_crafter`로 프롬프트 생성, `llm_client`로 질의, 응답 반환.
        - `_handle_user_confirmation(self, node_details: dict) -> str`: 사용자에게 확인/선택을 요청하는 노드 처리 (예: `input()`).
        - `_handle_code_execution(self, node_details: dict) -> dict`: 통계 코드 실행 노드 처리. `code_retriever`로 코드 검색, `safe_code_executor`로 실행.
        - `_update_analysis_parameters(self, new_params: dict)`: LLM 응답이나 사용자 결정에 따라 `analysis_parameters` 업데이트.
        - `_log_state_transition(self, from_node: str, to_node: str, reason: str)`: 상태 전이 로깅.

### 3. `core/workflow_manager.py`

- **기능**: `resources/workflow_graph.json` 파일을 로드, 파싱하고, 워크플로우 노드 및 전환 규칙에 대한 접근 인터페이스 제공.
- **역할**: 워크플로우 정의서 관리자.
- **주요 함수/클래스**:
    - `class WorkflowManager`:
        - `__init__(self, workflow_file_path: str)`: 워크플로우 JSON 파일 경로를 받아 내부적으로 로드.
        - `_workflow_definition: dict`: 로드된 JSON 데이터.
        - `get_node(self, node_id: str) -> dict | None`: 특정 ID의 노드 정보(description, subtasks, transitions 등) 반환. 없으면 None.
        - `get_initial_node_id(self) -> str`: 워크플로우 시작 노드 ID ("start") 반환.
        - `is_terminal_node(self, node_id: str) -> bool`: 해당 노드가 종료 노드인지 (더 이상 `transitions`이 없는지 등) 확인.

### 4. `core/decision_engine.py`

- **기능**: 현재 노드의 처리 결과 (LLM 응답, 사용자 입력, 코드 실행 결과 등)와 해당 노드의 `transitions` 규칙을 비교하여 다음으로 진행할 노드 ID를 결정.
- **역할**: 워크플로우 네비게이터.
- **주요 함수/클래스**:
    - `class DecisionEngine`:
        - `determine_next_node(self, current_node_details: dict, execution_outcome: any, user_response: str = None) -> str | None`:
            - `current_node_details`: 현재 노드의 전체 정보 (주로 `transitions` 필드 사용).
            - `execution_outcome`: 가장 최근 작업의 결과 (예: LLM이 생성한 텍스트, 코드 실행 성공/실패, 특정 값).
            - `user_response`: 사용자가 `input()` 등을 통해 제공한 응답.
            - `transitions` 배열을 순회하며 각 `condition`을 `execution_outcome` 및 `user_response`와 비교 평가.
            - 첫 번째로 만족하는 조건의 `next` 노드 ID를 반환. 만족하는 조건 없거나 에러 시 None 또는 예외 발생.
        - `_evaluate_condition(self, condition_string: str, outcome: any, user_input: str) -> bool`: 복잡한 조건 문자열을 파싱하고 평가하는 로직. (예: "사용자 '예' AND 결과값 > 0.5")

### 5. `core/context_manager.py` (신규)

- **기능**: LLM과의 상호작용 및 주요 분석 단계의 이력을 관리. 토큰 제한을 고려하여 이력을 요약하거나 필터링하여 LLM에 전달할 컨텍스트를 최적화.
- **역할**: Agent의 장기 기억 및 작업 메모리 관리자.
- **주요 함수/클래스**:
    - `class ContextManager`:
        - `__init__(self, llm_client: LLMClient, max_history_items: int = 20, summarization_trigger_count: int = 10, context_token_limit: int = 3000)`: LLM 클라이언트(요약용), 최대 저장할 상호작용 수, 요약 트리거 수, LLM 전달 컨텍스트 토큰 제한 설정.
        - `_interaction_history: list[dict]`: 각 상호작용(`{'role': 'user/assistant/system', 'content': '...', 'node_id': '...', 'timestamp': '...'}`) 저장.
        - `_summary_cache: str`: 가장 최근의 요약본.
        - `add_interaction(self, role: str, content: str, node_id: str)`: 새로운 상호작용을 이력에 추가. `summarization_trigger_count` 도달 시 자동 요약 고려.
        - `get_optimized_context(self, current_task_prompt: str, required_recent_interactions: int = 5) -> str`: 현재 작업 프롬프트와 함께 LLM에 전달할 최적화된 컨텍스트 문자열 반환.
            - 최근 `required_recent_interactions`는 포함.
            - 오래된 기록은 `_summary_cache`와 함께 조합하거나, 전체 기록이 `context_token_limit`을 넘지 않도록 조절.
            - 필요시 `_summarize_interactions()` 호출.
        - `_summarize_interactions(self, interactions_to_summarize: list[dict]) -> str`: 제공된 상호작용 목록을 `llm_client`를 사용해 요약. 결과를 `_summary_cache`에 업데이트.
        - `_prune_history(self)`: `max_history_items`를 초과하는 가장 오래된 상호작용(요약된 부분 제외) 제거.
        - `get_full_history_for_report(self) -> list[dict]`: 최종 보고서 생성을 위해 전체 원본 이력 반환.

### 6. `llm_services/llm_client.py`

- **기능**: 특정 LLM Provider(OpenAI, Gemini 등)의 API와 통신. 인증, 요청 생성, 응답 파싱, 기본 오류 처리 및 재시도 로직 포함.
- **역할**: LLM API 게이트웨이.
- **주요 함수/클래스**:
    - `class LLMClient`:
        - `__init__(self, api_key: str, model_name: str, provider_name: str, default_temperature: float = 0.5, max_retries: int = 3)`: API 키, 모델명, LLM 제공자 이름, 기본 온도값, 최대 재시도 횟수 설정.
        - `_session`: (선택적) `requests.Session` 또는 해당 SDK 클라이언트 인스턴스.
        - `generate_text(self, prompt: str, system_prompt: str = None, temperature: float = None, stop_sequences: list[str] = None) -> str`: 텍스트 생성을 위한 LLM API 호출.
        - `generate_chat_completion(self, messages: list[dict], temperature: float = None, stop_sequences: list[str] = None) -> str`: 채팅 형식 API 호출 (`messages`: `[{'role':'user', 'content':'...'}, ...]`).
        - `_handle_api_error(self, error_response)`: API 에러 공통 처리.

### 7. `llm_services/prompt_crafter.py`

- **기능**: `llm_services/prompts/` 디렉토리의 템플릿 파일을 기반으로, 현재 분석 컨텍스트와 노드 정보를 조합하여 LLM에 전달할 최종 프롬프트를 동적으로 생성.
- **역할**: 프롬프트 엔지니어링 및 조립 전문가.
- **주요 함수/클래스**:
    - `class PromptCrafter`:
        - `__init__(self, prompt_template_dir: str, workflow_data: dict = None)`: 프롬프트 템플릿 디렉토리 경로, (선택적) 전체 워크플로우 데이터 참조.
        - `_jinja_env`: (Jinja2 사용 시) `jinja2.Environment` 인스턴스.
        - `_load_template(self, template_name: str) -> jinja2.Template`: 지정된 이름의 템플릿 파일 로드.
        - `render_prompt(self, template_name: str, context_data: dict) -> str`: 템플릿과 컨텍스트 데이터를 결합하여 최종 프롬프트 문자열 생성.
        - `get_prompt_for_node(self, node_id: str, dynamic_data: dict, agent_context_summary: str = None) -> str`:
            - `node_id`에 매핑되는 템플릿 파일명 결정 (예: `f"stage_{node_id.split('-')[0]}/{node_id.replace('-', '_')}.md"`).
            - `dynamic_data`: 현재 노드 처리 위한 특정 데이터 (예: 사용자 요청 텍스트, 변수 목록).
            - `agent_context_summary`: `ContextManager`가 제공하는 요약된 이전 대화/작업 이력.
            - 이 모든 정보를 `context_data`로 만들어 `render_prompt` 호출.

### 8. `llm_services/prompts/` (디렉토리)

- **기능**: LLM에 전달될 프롬프트의 템플릿을 저장. Markdown(.md) 또는 Jinja2(.j2) 등의 텍스트 파일 형식 사용.
- **역할**: 프롬프트 내용과 구조를 코드와 분리하여 관리.
- **하위 구조 예시**:
    - `common/`: 여러 프롬프트에서 공통으로 사용될 수 있는 지시사항, 포맷팅 가이드라인 (예: `json_output_format.md`).
    - `stage_1_user_understanding/`: 워크플로우 1단계 관련 프롬프트 (예: `1_1_analyze_user_request.md`).
    - 각 파일은 변수 삽입 위치를 명시 (예: Jinja2의 `{{ variable_name }}`).

### 9. `data_processing/data_loader.py`

- **기능**: 지정된 경로의 데이터 파일(Tableau .hyper, CSV 등)을 로드하여 Pandas DataFrame으로 변환. 기본적인 데이터 정보(컬럼명, 추정 타입, 결측치 등) 추출 기능 제공.
- **역할**: 원시 데이터 접근 및 초기 탐색 정보 제공.
- **주요 함수/클래스**:
    - `class DataLoader`:
        - `load_data(self, file_path: str, file_type: str = None) -> pd.DataFrame`: 파일 경로와 타입에 따라 적절한 로더 사용. 파일 타입 미지정 시 확장자로 추론.
        - `_load_hyper(self, file_path: str) -> pd.DataFrame`: Tableau Hyper 파일 로드 (`pantab` 등 사용).
        - `_load_csv(self, file_path: str) -> pd.DataFrame`: CSV 파일 로드.
        - `get_data_profile(self, dataframe: pd.DataFrame, N_unique_threshold: int = 10) -> dict`: DataFrame의 각 컬럼에 대한 프로파일링 (데이터 타입, 결측치 수/비율, 고유값 수, 예시 값, (고유값 수가 적으면) 빈도수 상위 N개 등) 수행. Agent의 2-2, 2-3 단계 등에서 활용.

### 10. `rag_system/code_retriever.py`

- **기능**: 사용자의 분석 목적이나 명시된 통계 검정 방법에 가장 적합한 코드 스니펫을 `resources/code_snippets/` 에서 (또는 구축된 Vector DB에서) 검색.
- **역할**: RAG의 Retrieval. 코드 지식 베이스 접근.
- **주요 함수/클래스**:
    - `class CodeRetriever`:
        - `__init__(self, index_path_or_snippets_dir: str, embedding_model_name: str = None, top_k_results: int = 3)`: Vector DB 인덱스 경로 또는 코드 스니펫 원본 디렉토리, (필요시) 임베딩 모델, 반환할 결과 수 설정.
        - `_vector_db_client`: (Vector DB 사용 시) 클라이언트 인스턴스.
        - `_embedding_model`: (필요시) 텍스트 임베딩 생성 모델.
        - `find_relevant_code_snippets(self, query_description: str, required_variables: list[str] = None, language: str = "python") -> list[dict]`:
            - `query_description`: "독립표본 t-검정 수행 방법" 또는 "두 그룹 간 평균 비교 코드".
            - `required_variables`: (선택적) 코드 스니펫이 다루어야 할 변수명 정보.
            - 검색된 코드 스니펫의 내용, 출처(파일명), 관련성 점수 등을 담은 딕셔너리 리스트 반환.

### 11. `rag_system/code_indexer.py` (주로 초기 셋업 시 실행)

- **기능**: `resources/code_snippets/` 디렉토리의 모든 코드 파일을 읽어 텍스트 임베딩을 생성하고, 이를 Vector DB 또는 로컬 파일 기반 인덱스에 저장.
- **역할**: RAG 검색을 위한 지식 베이스 사전 구축.
- **주요 함수/클래스**:
    - `class CodeIndexer`:
        - `__init__(self, snippets_source_dir: str, target_index_path: str, embedding_model_name: str)`: 스니펫 소스 경로, 생성될 인덱스 저장 경로, 임베딩 모델 설정.
        - `_scan_snippet_files(self) -> list[str]`: 소스 디렉토리에서 모든 코드 파일 경로 스캔.
        - `_generate_embedding(self, code_text: str, metadata: dict) -> list[float]`: 코드 텍스트에 대한 임베딩 벡터 생성.
        - `build_and_save_index(self)`: 모든 스니펫 파일 처리 후 최종 인덱스 저장.

### 12. `code_execution/safe_code_executor.py`

- **기능**: RAG를 통해 검색된 문자열 형태의 Python 통계 코드를 가능한 안전한 방식으로 실행. 입력 데이터와 파라미터를 코드에 주입하고, 실행 결과(텍스트 출력, 변수 값, 이미지 데이터 등)를 캡처하여 반환.
- **역할**: 외부 코드의 동적 실행. **보안에 매우 민감한 부분.**
- **주요 함수/클래스**:
    - `class SafeCodeExecutor`:
        - `__init__(self, timeout_seconds: int = 30)`: 코드 실행 시간 제한 설정.
        - `execute_code(self, code_string: str, input_dataframe: pd.DataFrame, parameters: dict = None) -> dict`:
            - `code_string`: 실행할 Python 코드.
            - `input_dataframe`: 코드 내에서 `df` 등의 이름으로 참조될 Pandas DataFrame.
            - `parameters`: 코드 내 변수로 주입될 추가 파라미터.
            - **실행 방식 고려 사항**:
                1. **`restrictedpython`**: Python의 안전한 서브셋만 실행.
                2. **`subprocess` + `exec`**: 별도 프로세스에서 제한된 `globals`, `locals`와 함께 `exec` 실행. `stdout`, `stderr` 캡처. 자원 제한.
                3. **Docker 컨테이너**: 각 실행을 완전히 격리된 Docker 컨테이너 내부에서 수행 (가장 안전하나 무거움).
                4. **Pynbox, nsjail 등 샌드박싱 라이브러리**: 보다 정교한 샌드박싱 제공.
            - 반환 값: `{'stdout': str, 'stderr': str, 'result_variables': dict, 'generated_plots': list[bytes]}` 형태의 딕셔너리. `result_variables`는 코드 실행 후 특정 변수 값 추출.

### 13. `reporting/report_generator.py`

- **기능**: `LLMAgent`의 전체 분석 과정(선택된 노드, 주요 결정, LLM 상호작용 요약, 실행된 코드, 통계 결과)과 `ContextManager`의 이력을 종합하여 사용자 친화적인 최종 보고서(Markdown, HTML 등) 생성.
- **역할**: 분석 결과 및 과정의 최종 사용자 전달.
- **주요 함수/클래스**:
    - `class ReportGenerator`:
        - `__init__(self, output_directory: str, report_format: str = "md")`: 결과 보고서 저장 디렉토리, 기본 포맷 설정.
        - `generate_report(self, agent_final_state: dict, full_interaction_history: list[dict], data_profile: dict, workflow_graph_info: dict) -> str`:
            - `agent_final_state`: Agent의 `analysis_parameters`, 최종 데이터 요약 등.
            - `full_interaction_history`: `ContextManager`가 제공하는 전체 이력.
            - 보고서 내용을 구성하고 파일로 저장 후, 파일 경로 반환.
        - `_format_interaction(self, interaction_log: dict) -> str`: 단일 상호작용 로그를 보고서 형식에 맞게 변환.
        - `_format_code_execution(self, code_details: dict) -> str`: 코드 실행 결과(stdout, plot 등)를 보고서 형식으로 변환.

### 14. `config/settings.py`

- **기능**: 애플리케이션 전반의 설정값을 관리. 환경 변수에서 로드하거나 기본값을 가질 수 있음.
- **역할**: 설정 중앙화.
- **변수 예시**:
    - `LLM_PROVIDER = "openai"` (또는 "gemini")
    - `OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")`
    - `GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")`
    - `LLM_MODEL_NAME = "gpt-4o"`
    - `WORKFLOW_FILE_PATH = "resources/workflow_graph.json"`
    - `CODE_SNIPPETS_DIR = "resources/code_snippets/"`
    - `RAG_INDEX_PATH = "resources/rag_index/code_snippets.index"` (FAISS 등)
    - `EMBEDDING_MODEL_NAME = "text-embedding-ada-002"` (OpenAI) 또는 다른 Sentence Transformer 모델
    - `INPUT_DATA_DEFAULT_DIR = "input_data/"`
    - `OUTPUT_RESULTS_DIR = "output_results/"`
    - `LOG_LEVEL = "INFO"`

### 13. `.env`

- **기능**: 실제 API 키 및 기타 민감한 설정값을 저장합니다. **이 파일은 절대 Git에 커밋하면 안 됩니다.**
- **내용 예시**: