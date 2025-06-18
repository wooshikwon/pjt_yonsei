# 개발 작업 로그 (Factoring Log)

이 문서는 System Implementation Agent가 수행하는 모든 파일 생성, 수정, 삭제 작업을 기록합니다.
---
- 2024-07-25: `src/components/context.py` 파일 생성. `Context` 클래스 및 관련 메서드 구현.
- 2024-07-25: `src/components/code_executor.py` 파일 생성. `CodeExecutor` 클래스 및 `run` 메서드 구현.
- 2024-07-25: `src/components/rag_retriever.py` 파일 생성. `RAGRetriever` 클래스 및 관련 메서드 구현.
- 2024-07-25: `src/prompts/system_prompts.py` 파일 생성. 시스템 프롬프트 상수 4개 추가.
- 2024-07-25: `src/agent.py` 파일 생성. `Agent` 클래스 및 OpenAI API 연동 메서드 구현.
- 2024-07-25: `src/components/code_executor.py` 수정. `run` 메서드가 전역 변수(e.g., `df`)를 받을 수 있도록 변경.
- 2024-07-25: `src/main.py` 파일 생성. Typer CLI 기반의 전체 분석 파이프라인(Orchestrator) 구현.
- 2024-07-25: `Dockerfile` 및 `.dockerignore` 파일 생성. Poetry 프로젝트의 컨테이너화를 위함.
- 2024-07-25: `tests/test_code_executor.py` 파일 생성. `CodeExecutor`의 성공, 실패 및 `global_vars` 사용 사례에 대한 단위 테스트 3개 작성.
- 2024-07-25: `tests/test_rag_retriever.py` 파일 생성. `tmp_path`를 사용한 `RAGRetriever`의 인덱스 빌드 및 쿼리 기능에 대한 단위 테스트 작성.
- 2024-07-25: `src/components/rag_retriever.py` 수정. FAISS 인덱스 차원을 LlamaIndex 기본 임베딩 모델에 맞게 1536으로 변경.
- 2024-07-25: `src/components/rag_retriever.py` 재수정. API 의존성 제거를 위해 로컬 임베딩 모델(`ko-sroberta-multitask`)을 사용하도록 변경.
- 2024-07-25: `resources/knowledge_base`에 `.gitkeep` 파일 추가 및 `RAGRetriever` 안정성 강화(인덱스 로딩 로직 개선, 빈 문서 처리).
- 2024-07-25: 아키텍처 변경: RAG는 로컬 임베딩 모델(`ko-sroberta-multitask`)만 사용하도록 `RAGRetriever`를 리팩토링하고 `BLUEPRINT.md`에 반영.
- 2024-07-25: `tests/test_agent.py` 파일 생성. `mocker`를 사용하여 `_call_api`를 모의 처리하고 `generate_analysis_plan` 메서드 테스트.
- 2024-07-25: `pyproject.toml`에 `python-dotenv` 라이브러리 추가.
- 2024-07-25: `env.example` 파일 생성. RAG 및 LLM 설정을 위한 환경 변수 템플릿 정의.
- 2024-07-25: `output_data/rag_storage` 및 `resources/vector_store` 디렉토리 삭제. RAG 인덱스 저장소 구조 정리.
- 2024-07-25: `src/components/rag_retriever.py` 리팩토링. FAISS 인덱스 로딩/저장 로직 수정 및 `rebuild` 옵션 추가.
- 2024-07-25: `src/main.py` 수정. `.env` 파일 로드 및 `USE_RAG`, `REBUILD_VECTOR_STORE` 환경 변수를 파이프라인에 적용.
- 2024-07-25: `BLUEPRINT.md` 업데이트. `.env` 설정, RAG 제어 옵션, `resources/rag_index` 디렉토리 구조 변경 사항 반영.
- 2024-07-25: **테스트 오류 진단 및 해결**: RAG 테스트 실패 원인이 파일명 불일치임을 발견. `FaissVectorStore`가 `default__vector_store.json`로 저장하는데 테스트에서 `vector_store.json`을 찾고 있었음. 
- 2024-07-25: `tests/test_rag_retriever.py` 수정. 디버깅용 파일 목록 출력 기능 추가 및 올바른 파일명(`default__vector_store.json`) 사용으로 변경.
- 2024-07-25: **전체 테스트 통과 확인**: pytest 실행 결과 모든 8개 테스트 성공. RAG 인덱스 생성, 로딩, 재빌드 기능이 정상적으로 동작함을 검증 완료.
- 2024-07-25: **모듈 경로 문제 해결**: `src/main.py` 실행 시 `ModuleNotFoundError` 해결. `python -m src.main` 방식으로 실행하도록 수정.
- 2024-07-25: `src/agent.py`에 `_clean_code_response()` 메서드 추가. LLM이 생성하는 코드에서 markdown 백틱 제거하여 `SyntaxError` 방지.
- 2024-07-25: **🎉 전체 시스템 동작 성공**: RAG 컨텍스트 강화부터 최종 보고서 생성까지 모든 파이프라인이 정상 동작함을 확인. `normal_sales.csv` 데이터로 A팀과 B팀 성과 차이 분석 완료.
- 2024-07-25: **로깅 시스템 구축**: `src/utils/logger.py` 생성으로 날짜별 상세 로그 파일 저장 및 터미널 출력 간소화 시스템 구현.
- 2024-07-25: `src/main.py` 로깅 시스템 적용. 단계별 진행 상황만 터미널에 표시하고 상세 정보는 `logs/` 디렉토리에 저장.
- 2024-07-25: `src/components/rag_retriever.py` 로깅 개선. print 출력을 파일 로깅으로 대체하여 터미널 출력 정리.
- 2024-07-25: **예제 데이터 생성**: `team_sales_performance.csv`, `customer_survey.csv` 통계 검정용 샘플 데이터 생성.
- 2024-07-25: **✅ 로깅 시스템 동작 확인**: 터미널은 단계별 진행상황만 표시, 상세 로그는 `logs/analysis_YYYYMMDD.log`에 자동 저장. 깔끔한 최종 보고서 출력 완료.
- 2024-07-25: **터미널 출력 완전 정리**: HuggingFace tokenizers, scipy 경고, LlamaIndex 로딩 메시지 등 모든 라이브러리 출력을 숨김 처리.
- 2024-07-25: `src/main.py`, `src/components/code_executor.py`, `src/components/rag_retriever.py`에 경고 및 출력 억제 코드 추가.
- 2024-07-25: **🎯 최종 완성**: 터미널은 단계별 진행 상황과 최종 보고서만 깔끔하게 표시. 모든 기술적 메시지는 로그 파일로 분리 완료.
- 2024-07-25: **🧹 코드 정리 작업**: deprecated된 `query()` 메서드 제거, 불필요한 .DS_Store 파일 삭제, 미사용 환경변수 `LLM_PROVIDER` 제거.
- 2024-07-25: **⚙️ 경고 설정 통합**: `src/utils/warnings_config.py` 생성으로 중복된 경고 및 로깅 설정을 통합. `main.py`와 `rag_retriever.py`에서 중복 코드 제거.
- 2024-07-25: **📝 테스트 주석 수정**: `tests/test_rag_retriever.py`의 부정확한 OpenAI API 키 필요 언급을 로컬 임베딩 모델 사용으로 수정.
- 2024-07-25: **🔧 warnings_config 모듈 완성**: `suppress_warnings()` 컨텍스트 매니저 추가. `CodeExecutor`에서 통합 모듈 사용하도록 리팩토링.
- 2024-07-25: **✅ 전체 테스트 통과**: 모든 8개 테스트 성공. 통합된 경고 설정으로 시스템 전체의 일관성 확보.
- 2024-07-25: **📊 복합 테스트 데이터셋 생성**: 5가지 통계 분석 유형(ANOVA, Linear Regression, Logistic Regression, Correlation, 비율검정)을 위한 테스트 데이터 생성.
- 2024-07-25: **🧪 통합 테스트 시나리오 구축**: `tests/integration_test_scenarios.md` 및 개별 시나리오 파일들로 시스템 강건성 평가 체계 완성.
- 2024-07-25: **🚫 시각화 완전 차단**: matplotlib 백엔드 'Agg' 강제 설정 및 시스템 프롬프트에 시각화 금지 가이드라인 추가. 수치 중심 분석으로 통일.
- 2024-07-25: **🔄 환경변수 독립화**: `USE_RAG`와 `REBUILD_VECTOR_STORE`를 독립적으로 작동하도록 리팩토링. 벡터 스토어 관리와 RAG 사용 분리.
- 2024-07-25: **💬 사용자 안내 개선**: RAG 인덱스 부재 시 터미널에 명확한 해결 방법 안내 메시지 추가. 사용자 경험 향상.
- 2024-07-25: **🔧 환경변수 로딩 수정**: `load_dotenv(override=True)` 추가로 .env 파일 변경사항 실시간 반영. 환경변수 독립 동작 완전 구현.
- 2024-07-25: **✅ 독립 환경변수 테스트 완료**: `USE_RAG=False, REBUILD_VECTOR_STORE=True` 시나리오 성공. 벡터 스토어 재구축과 RAG 사용이 완전히 분리 작동 확인.
- 2024-07-25: **🐳 Docker 환경 완전 개선**: Dockerfile, docker-compose.yml, .dockerignore 전면 리팩토링. 보안 강화된 환경변수 처리 및 개발/프로덕션 분리.
- 2024-07-25: **🔒 보안 강화**: .env 파일 volume mount 방식으로 API 키 유출 방지. 이미지에 민감정보 포함되지 않는 안전한 구조 구축.
- 2024-07-25: **🚀 자동화된 검증 스크립트 구축**: `tests/validation.py` 생성. `qa.json`의 모든 테스트 케이스를 자동으로 실행하고, 실제 결과와 기대 결과를 비교할 수 있는 `validation_result.json` 생성.
- 2024-07-25: **🛡️ 검증 스크립트 강건성 강화**: `validation.py` 리팩토링. 테스트 실패 시에도 중단되지 않고, 유연한 결과 추출 및 가독성 높은 출력으로 개선.
- 2024-07-25: **🚀 Poetry 환경 호환성 확보**: `validation.py`가 `poetry run` 가상환경에서 실행되도록 수정. `ModuleNotFoundError` 및 `pyenv` 버전 충돌 문제 해결.
- 2024-07-25: **📝 표준 .gitignore 생성**: 일반적인 Python 프로젝트의 .gitignore 파일을 생성하고, output_data/를 포함하여 불필요한 파일들을 무시하도록 설정.
- 2024-07-25: **🧹 .dockerignore 정리**: .gitignore와 중복되는 내용을 정리하여 Docker 이미지 빌드에 필요한 최소한의 파일만 포함하도록 수정.
- 2024-07-25: **📖 포괄적 README.md 작성**: 프로젝트 개요, 설치 가이드, 사용 예시, 문제 해결 등 포함.
- 2024-07-25: **🔨 지식 베이스 빌드 스크립트 분리**: `src/build_knowledge_base.py` 생성. 분석 실행과 RAG 인덱스 관리를 분리하여 시스템 모듈화.
- 2024-07-25: **📝 모든 문서 업데이트**: `README.md`, `factoring_log.md`, `BLUEPRINT.md`에 빌드 스크립트 분리 내용 반영.
- 2024-07-25: **🔧 .gitignore 수정**: `output_data/`와 `logs/` 디렉토리 자체는 유지하되, 그 안의 내용만 무시하도록 규칙 수정. (.gitkeep 추가)
- 2024-07-26: **🚀 QA 검증 자동화 스크립트 생성**: `run_qa.py`를 생성하여 `qa.json`의 모든 테스트 케이스를 자동으로 실행하고, 실제 결과와 기대 결과를 비교할 수 있는 `qa_result.json`을 생성하는 기능을 구현함.
- 2024-07-26: **📦 RAG 인덱스 디렉토리 생성**: `BLUEPRINT.md` 설계에 따라 RAG 벡터 인덱스를 저장할 `resources/rag_index` 디렉토리를 생성하고, Git 추적을 위한 `.gitkeep` 파일을 추가함.
- 2024-07-26: **📚 RAG 지식 베이스 구축**: `input_data/data_files/`에 있는 7개의 표준 데이터셋(boston_housing, Iris, mpg, pima-indians-diabetes, teachingratings, tips, titanic)에 대한 설명 및 컬럼 정의를 담은 마크다운 문서를 `resources/knowledge_base/`에 생성함.
- 2024-07-26: **🧠 QA 시나리오 고도화**: `tests/qa.json`의 모든 질문에서 명시적인 통계 용어(ANOVA, T-검정, 유의수준 등)를 제거하고, 사용자의 의도를 파악해야 하는 자연어 질문으로 재구성하여 시스템의 LLM 추론 능력 테스트 케이스를 강화함.
- 2024-07-26: **🔧 의존성 오류 해결**: `warnings_config` 모듈에서 사용되는 `matplotlib` 라이브러리가 `pyproject.toml`에 누락되어 발생한 `ModuleNotFoundError`를 해결하기 위해 `poetry add matplotlib` 명령어로 의존성을 추가함.
- 2024-07-26: **🚫 시각화 차단 로직 개선**: '시각화를 막기 위해 시각화 라이브러리를 설치'하는 역설적인 구조를 개선. `warnings_config.py`에서 `matplotlib` 관련 코드를 삭제하고, `poetry remove`를 통해 의존성을 완전 제거하여 보다 근본적인 방식으로 시각화 실행을 차단함.
- 2024-07-26: **🚀 QA 검증 자동화 스크립트 재생성**: 사용자의 요청에 따라 `run_qa.py` 파일을 재생성함. 이 스크립트는 `tests/qa.json`의 모든 테스트 케이스를 자동으로 실행하고, 그 결과를 `qa_result.json`으로 저장하는 기능을 수행함.
- 2024-07-26: **🛡️ 시각화 차단 정책 재변경**: 논의 끝에, 라이브러리 부재로 인한 `ModuleNotFoundError`보다 `matplotlib.use('Agg')`로 명시적 백엔드를 설정하는 것이 더 안정적이고 의도가 명확하다고 판단함. `warnings_config.py` 코드를 원복하고 `matplotlib` 의존성을 다시 추가하는 방향으로 정책을 재수정함.
- 2024-07-26: **🚚 QA 스크립트 경로 수정**: `run_qa.py`를 `tests/` 디렉토리로 이동하고, 결과 파일(`qa_result.json`)도 동일한 `tests/` 디렉토리에 저장되도록 스크립트 내부 경로 로직을 수정함.
- 2024-07-26: **📖 README.md 문서 현행화**: RAG 인덱스 빌드 방법(`embedder.py` 실행)을 추가하고, '사용 예시' 및 '샘플 데이터' 섹션을 현재 보유한 데이터셋과 자연어 질의에 맞게 전면 수정하여 문서의 정확성과 유용성을 높임.

# Refactoring Log

## 2025-01-21

### 🧹 Docker 설정 단순화 요청 대응
- **작업**: Docker 복잡성 제거 작업
- **파일 수정**: `docker-compose.yml` - 개발환경과 API 서버 관련 서비스 제거
- **파일 삭제**: `Dockerfile.dev` - 개발환경용 Dockerfile 삭제
- **설정**: 기본 분석용 서비스만 유지하여 설정 단순화
- **사용법 업데이트**: `docker-compose run --rm statistical-analyzer --file <filename> --request "<request>"`

### 🔒 Docker 보안 검증 완료
- **확인**: API 키 노출 방지 검증 완료
- **보안 조치**: .env 파일이 이미지에 포함되지 않고 런타임에만 마운트됨 확인
- **검증 결과**: 이미지 빌드/공유 시 민감정보 노출 위험 없음

### 🚀 환경 변수 독립성 확보
- **개선**: `USE_RAG`와 `REBUILD_VECTOR_STORE` 독립 작동 구현
- **Step 0**: 벡터 저장소 관리 (선택적)
- **Step 1**: RAG 사용 여부 (독립적)
- **사용자 안내**: RAG 인덱스 부재 시 `REBUILD_VECTOR_STORE=True` 안내 메시지 추가
- **실시간 반영**: `load_dotenv(override=True)`로 .env 파일 변경 즉시 반영

### 🎨 시각화 방지 시스템 구축
- **기술적 차단**: `matplotlib.use('Agg')` 설정으로 GUI 백엔드 비활성화
- **AI 가이드라인**: 시스템 프롬프트에 시각화 제한 지침 추가
- **효과**: 예상치 못한 이미지 팝업 완전 차단

### 🧽 코드 정리 및 최적화
- **deprecated 메서드 제거**: `RAGRetriever.query()` 메서드 삭제
- **파일 정리**: .DS_Store 파일 삭제
- **환경변수 정리**: 사용하지 않는 `LLM_PROVIDER` 변수 제거
- **경고 억제 통합**: `src/utils/warnings_config.py` 모듈로 통합
- **테스트 주석 수정**: OpenAI API 키 불필요 명시

### 📊 포괄적 테스트 시나리오 생성
- **5가지 분석 유형**: ANOVA, 선형회귀, 로지스틱회귀, 상관분석, Z-검정
- **데이터셋 생성**: 각 통계 분석에 최적화된 샘플 데이터 제작
- **시나리오 문서**: `tests/integration_test_scenarios.md` 및 개별 시나리오 파일 생성
- **다양한 표현**: 자연어 요청의 다양성 테스트를 위한 여러 표현법 포함

### 🔧 터미널 UX 대폭 개선
- **진행 표시**: 단계별 이모지와 함께 명확한 진행 상황 표시
- **로그 분리**: 상세 정보는 날짜별 로그 파일에, 터미널은 핵심만
- **경고 억제**: 모든 라이브러리 경고/정보 메시지 완전 숨김
- **전문적 출력**: 비즈니스 사용자 친화적 인터페이스 완성

### 📖 포괄적 README.md 작성
- **작업**: 빈 README.md 파일을 완전한 프로젝트 문서로 개선
- **내용**: 프로젝트 개요, 주요 특징, 지원 분석 유형, 설치 가이드 포함
- **설치 옵션**: Poetry와 Docker 두 가지 방법 모두 상세 설명
- **사용 예시**: 7개 샘플 데이터와 4가지 분석 유형별 실제 명령어 제공
- **문제 해결**: 자주 발생하는 4가지 문제와 해결책 포함
- **Github 연동**: 실제 저장소 경로와 이슈 트래킹 링크 포함