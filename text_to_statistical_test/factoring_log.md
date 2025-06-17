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