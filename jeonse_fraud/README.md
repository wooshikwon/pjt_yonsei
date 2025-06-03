jeonse_fraud/
├── README.md                    # 프로젝트 개요, 디렉토리 구조, 설정 및 실행 방법 안내
├── docker-compose.yml           # (선택 사항) 다중 컨테이너 Docker Compose 설정 (예: 별도 Vector DB 서비스)
├── Dockerfile                   # PoC 애플리케이션 실행을 위한 Docker 이미지 빌드 설정
│
├── config/                      # 설정 파일 디렉토리
│   ├── settings.yaml            # API 키, 모델명, 파일 경로, 임계값 등 주요 설정
│   └── logging_config.yaml      # 로깅 형식 및 레벨 설정 (Python logging 모듈 연동)
│
├── data/                        # 모든 데이터 관련 파일 루트 디렉토리
│   ├── input/                   # PoC 실행을 위한 테스트 케이스별 입력 데이터
│   │   └── test_case_001/       # 예시 테스트 케이스
│   │       ├── property_register_input.json     # 등기부등본 상세 내용 (필수, 구조화된 JSON)
│   │       ├── building_ledger_api_mock.json  # 건축물대장 API 응답 모의 데이터 (JSON)
│   │       ├── transaction_price_api_mock.json# 실거래가 API 응답 모의 데이터 (JSON)
│   │       └── contract_info_input.json       # 계약 조건 관련 정보 (JSON)
│   │   └── test_case_002/       # 다른 테스트 케이스
│   │       └── ...
│   ├── rag_sources/             # RAG Vector DB 구축을 위한 원천 문서 (텍스트 파일 등)
│   │   ├── legal_cases/         # 판례 (예: case_001.txt)
│   │   ├── statutes/            # 법령 (예: housing_act.txt)
│   │   ├── fraud_examples/      # 사기 사례 분석 (예: kkangtong_analysis.txt)
│   │   └── prevention_guides/   # 예방 가이드 (예: jeonse_checklist.txt)
│   └── vector_db/               # 생성된 로컬 Vector DB 파일 저장소 (예: ChromaDB, FAISS 인덱스)
│       └── (db_files_or_directory_here)
│
├── docs/                        # 프로젝트 관련 문서 (설계, 평가 기준 등)
│   └── evaluation_framework.md  # LLM 평가 기준 상세 및 PoC 성공 기준 정의
│
├── logs/                        # 애플리케이션 실행 로그 저장 디렉토리
│   └── app_YYYY-MM-DD.log       # 날짜별 로그 파일
│
├── notebooks/                   # (선택 사항) 데이터 탐색, 프로토타이핑용 Jupyter Notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_rag_source_processing.ipynb
│   └── 03_prompt_engineering_tests.ipynb
│
├── scripts/                     # 유틸리티 및 준비 스크립트
│   └── build_vector_db.py       # data/rag_sources/ 에서 Vector DB 구축 및 data/vector_db/ 에 저장
│
├── src/                         # 애플리케이션 핵심 소스 코드 (Python 패키지)
│   ├── __init__.py
│   ├── main.py                  # PoC 파이프라인 실행 메인 스크립트 (CLI 인자 처리)
│   ├── config_loader.py         # config/settings.yaml 및 logging_config.yaml 로드 모듈
│   ├── data_ingestion.py        # data/input/ 에서 특정 테스트 케이스 데이터 로드 및 기본 검증
│   ├── external_api_handler.py  # (모의) 외부 API 연동 처리 (부동산 실거래가, 건축물대장)
│   ├── rag_retriever.py         # data/vector_db/ 에 저장된 Vector DB 질의 및 관련 문서 검색
│   ├── llm_service.py           # LLM API 호출, 프롬프트 포맷팅, 응답 파싱 등 LLM 연동 핵심 로직
│   ├── risk_assessment_engine.py # llm_service 및 rag_retriever를 활용한 위험 평가 로직
│   ├── solution_advisor.py      # llm_service 및 rag_retriever를 활용한 해결책(특약 등) 제안 로직
│   └── report_generator.py      # 최종 분석 보고서 파일 생성 (Markdown 또는 기타 형식)
│
├── prompts/                     # LLM 프롬프트 템플릿 저장 디렉토리 (Markdown 또는 텍스트 파일)
│   ├── system_role_prompt.md    # LLM의 기본 역할/성격 정의 (예: "당신은 전세사기 위험 분석 전문가입니다...")
│   ├── risk_assessment_chain/   # 위험 분석을 위한 단계별 또는 연쇄 프롬프트
│   │   ├── 01_property_register_analysis_prompt.md
│   │   ├── 02_financial_leverage_analysis_prompt.md
│   │   └── 03_overall_risk_synthesis_prompt.md
│   └── advisory_generation_prompt.md # 조언 및 특약 조건 생성용 프롬프트
│
├── reports/                     # 생성된 최종 분석 보고서 저장 디렉토리
│   ├── test_case_001_report.md
│   └── ...
│
└── requirements.txt       

## 실행 방법 (PoC)

1. 의존성 설치

```bash
pip install -r requirements.txt
```

2. 환경 변수 설정 (예: .env 파일)

```
OPENAI_API_KEY=sk-...
DATAPORTAL_API_KEY=...
REAL_ESTATE_API_KEY=...
```

3. 입력 데이터 준비

- `data/input/test_case_001/property_register_input.json` (등기부등본)
- `data/input/test_case_001/contract_info_input.json` (계약 정보)

4. 파이프라인 실행

```bash
python -m src.main --test_case_id test_case_001
```

5. 결과 보고서 확인

- `reports/` 디렉토리에서 생성된 Markdown 파일 확인

## 주요 기능
- 등기부/계약정보 입력 → 외부 API(건축물대장, 실거래가) 조회
- 판례/법령/사기사례 등 RAG 기반 정보 검색
- LLM 기반 위험 평가 및 특약/주의사항 제안
- 결과 보고서 자동 생성       