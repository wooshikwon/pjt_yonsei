mtp_llm_finetuning/
├── Dockerfile                    # Docker 이미지 빌드 설정 파일
├── .dockerignore                 # Docker 이미지 빌드 시 제외할 파일 및 디렉토리 목록
├── pyproject.toml                # Poetry 프로젝트 설정 및 의존성 관리 파일
├── poetry.lock                   # Poetry 의존성 잠금 파일 (poetry install/lock 시 자동 생성)
├── README.md                     # 프로젝트 설명 및 실행 방법
├── requirements.txt              # (Poetry 사용 시 선택 사항, poetry export로 생성 가능)
│
├── configs/                      # 설정 파일 디렉토리
│   └── default_config.yaml       # 기본 학습 설정 (경로, 하이퍼파라미터 등)
│   └── experiment_XYZ.yaml       # 특정 실험을 위한 설정 파일
│
├── data/                         # 데이터 디렉토리 (Docker 빌드 시 제외, 볼륨 마운트 권장)
│   ├── oasst1/
│   └── processed/
│       └── oasst_pairs.jsonl
│
├── models_MTP/                   # MTP 모델 관련 파일 (Docker 빌드 시 제외, 볼륨 마운트 권장)
│   ├── 7B_1T_4/
│   │   ├── params.json
│   │   ├── consolidated.00.pth   # (또는 여러 .pth 파일)
│   │   └── ...
│   └── tokenizer/
│       └── tokenizer.model
│
├── saved_models/                 # 학습된 모델 및 체크포인트 저장 위치 (Docker 빌드 시 제외, 볼륨 마운트 권장)
│   ├── critic_experiment_A/
│   └── mtp_finetuned_experiment_A/
│
├── src/                          # 소스 코드 디렉토리 (Docker 이미지에 복사)
│   ├── __init__.py
│   ├── model_def/
│   │   ├── __init__.py
│   │   ├── mtp_transformer.py
│   │   └── critic_head.py
│   ├── data_utils/
│   │   ├── __init__.py
│   │   └── oasst_processor.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── critic_trainer.py
│   │   └── mtp_trainer.py
│   │   └── utils.py
│   ├── main.py
│   └── load_utils.py
│
├── scripts/                      # 쉘 스크립트 (실행 편의용)
│   ├── download_mtp_model.sh
│   ├── train_critic.sh
│   └── finetune_llm.sh
│
└── logs/                         # 학습 로그 저장 디렉토리 (Docker 빌드 시 제외, 볼륨 마운트 권장)
    ├── critic_training.log
    └── mtp_finetuning.log