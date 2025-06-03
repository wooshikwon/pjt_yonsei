jeonse_fraud_poc/
├── README.md                    # 프로젝트 개요, 디렉토리 구조, 설정 및 실행 방법 안내
├── docker-compose.yml           # (선택 사항) Docker Compose 설정 파일
├── Dockerfile                   # PoC 애플리케이션 실행을 위한 Docker 이미지 빌드 설정 파일
│
├── scripts/                     # 유틸리티 및 준비 스크립트
│   └── build_vector_db.py       # rag_data/의 원천 문서로 Vector DB를 구축하는 스크립트
│
├── src/                         # 애플리케이션 핵심 소스 코드
│   ├── main.py                  # 전체 PoC 프로세스를 관장하는 메인 스크립트
│   ├── llm_handler.py           # LLM API 호출 및 응답 처리 모듈
│   ├── rag_handler.py           # RAG 시스템 (Vector DB) 질의 및 결과 처리 모듈
│   ├── data_processor.py        # 입력 데이터(수동, API 시뮬레이션) 처리 및 정제 모듈
│   ├── report_generator.py      # 최종 분석 보고서 생성 모듈
│   └── utils.py                 # 기타 유틸리티 함수 모듈
│
├── prompts/                     # LLM에 사용될 프롬프트 템플릿 저장 디렉토리
│   ├── risk_assessment_prompt.md
│   └── solution_generation_prompt.md
│
├── rag_data/                    # RAG 시스템의 Vector DB 구축을 위한 원천 데이터
│   │                            # (판례, 법령, 사기 사례, 예방 가이드라인 등)
│   ├── legal_cases/
│   ├── statutes/
│   ├── fraud_examples/
│   └── prevention_guides/
│
├── vector_db/                   # 생성된 로컬 Vector DB 파일들이 저장되는 디렉토리
│   │                            # (예: ChromaDB 데이터 디렉토리, FAISS 인덱스 파일 등)
│   └── (db_files_or_directory_here)
│
├── input_data/                  # PoC 실행을 위한 테스트 케이스별 입력 데이터
│   ├── test_case_001/
│   │   ├── property_register_details.json
│   │   ├── building_ledger_api_mock.json
│   │   ├── transaction_price_api_mock.json
│   │   └── contract_details.json
│   └── ...
│
├── output_reports/              # LLM이 생성한 최종 분석 보고서 저장 디렉토리
│   ├── test_case_001_report.md
│   └── ...
│
├── evaluation_criteria/         # 평가 기준 관련 문서
│   └── detailed_evaluation_criteria_checklist.md
│
└── requirements.txt             # 프로젝트 실행에 필요한 Python 라이브러리 및 의존성 목록