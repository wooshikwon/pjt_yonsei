# Text-To-Statistical-Test(TTST)

RAG 기반 Agentic AI 통계 분석 시스템

## 개요

Text-to-Statistical-Test는 자연어 입력을 통해 자동화된 통계 분석을 제공하는 시스템입니다. RAG(Retrieval Augmented Generation)와 Agentic LLM을 활용하여 사용자와의 대화를 통해 데이터를 분석하고, 최적의 통계 검정을 수행하며, 해석 가능한 보고서를 생성합니다.

## 주요 기능

- **8단계 워크플로우**: 데이터 선택부터 보고서 생성까지 체계적인 분석 과정
- **Agentic AI**: LLM Agent가 자율적으로 분석 방법을 결정하고 실행
- **RAG 시스템**: 도메인 지식과 통계 방법론을 활용한 지능적 분석
- **다양한 통계 검정**: t-검정, ANOVA, 회귀분석, 비모수 검정 등 포괄적 지원
- **자동 보고서 생성**: HTML, PDF, JSON 등 다양한 형식의 결과 제공
- **대화형 인터페이스**: 사용자 친화적인 CLI 인터페이스

## 🚀 빠른 시작

### 자동 설정 스크립트 사용

```bash
# 저장소 클론
git clone <repository-url>
cd text-to-statistical-test

# 자동 설정 실행
chmod +x setup.sh
./setup.sh
```

자동 설정 스크립트가 다음을 수행합니다:
- 환경변수 파일 (.env) 생성
- API 키 설정 도움
- Poetry 또는 Docker 설치/설정
- 의존성 설치

### 수동 설정

## 설치 방법

### 필수 요구사항

- Python 3.11 이상
- Poetry (의존성 관리) 또는 Docker
- OpenAI 또는 Anthropic API 키

### 1. 레포지토리 클론

```bash
git clone <repository-url>
cd text-to-statistical-test
```

### 2. 환경변수 설정 (중요! 🔐)

```bash
# 환경변수 파일 복사
cp env.example .env

# .env 파일 편집하여 API 키 설정
# 텍스트 에디터로 .env 파일을 열고 다음과 같이 설정:
```

```bash
# 필수: OpenAI API 키 (최소 하나는 필요)
OPENAI_API_KEY=sk-your-actual-openai-api-key-here

# 선택사항: Anthropic API 키
ANTHROPIC_API_KEY=your-actual-anthropic-api-key-here
```

### 3-A. Poetry 설치 방법

```bash
# Poetry 설치 (없는 경우)
curl -sSL https://install.python-poetry.org | python3 -

# 의존성 설치
poetry install

# 실행
poetry run python main.py --help
```

### 3-B. Docker 설치 방법

```bash
# Docker 이미지 빌드
docker build -t text-to-statistical-test .

# 환경변수 파일과 함께 실행
docker run --env-file .env \
           -v $(pwd)/input_data:/app/input_data \
           -v $(pwd)/output_data:/app/output_data \
           text-to-statistical-test

# 또는 Docker Compose 사용
docker-compose up
```

## 사용 방법

### 기본 실행

```bash
# Poetry 환경에서 실행
poetry run python main.py

# Docker 환경에서 실행
docker run --env-file .env text-to-statistical-test python main.py --interactive
```

### 명령행 옵션

```bash
# 특정 파일로 시작
python main.py --file data.csv

# 특정 단계부터 시작
python main.py --stage 3

# 비대화형 모드
python main.py --non-interactive

# 디버그 모드
python main.py --debug

# 출력 형식 지정
python main.py --export-format pdf

# 특정 단계 건너뛰기
python main.py --skip-stages 2,4

# 도움말
python main.py --help
```

### Docker 고급 사용법

```bash
# 방법 1: 환경변수 파일 사용 (권장)
docker run --env-file .env text-to-statistical-test

# 방법 2: 개별 환경변수 전달
docker run -e OPENAI_API_KEY="your-key" text-to-statistical-test

# 방법 3: 대화형 모드
docker run -it --env-file .env text-to-statistical-test python main.py --interactive

# 방법 4: 볼륨 마운트로 데이터 연결
docker run --env-file .env \
           -v $(pwd)/input_data:/app/input_data \
           -v $(pwd)/output_data:/app/output_data \
           text-to-statistical-test
```

## 🔐 보안 가이드

### API 키 보안

**⚠️ 중요: API 키 보안을 위해 반드시 지켜야 할 사항들**

1. **절대 API 키를 코드에 하드코딩하지 마세요**
2. **`.env` 파일을 Git에 커밋하지 마세요** (이미 `.gitignore`에 포함됨)
3. **Docker 이미지를 공유할 때 API 키가 포함되지 않도록 주의하세요**
4. **환경변수나 시크릿 관리 서비스를 사용하세요**

### 환경변수 우선순위

시스템은 다음 순서로 환경변수를 로드합니다:

1. **시스템 환경변수** (최우선)
2. **`.env` 파일** 
3. **기본값**

### 프로덕션 배포 시 권장사항

#### 1. 환경변수 직접 설정
```bash
# Linux/macOS
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# Windows
set OPENAI_API_KEY=your-key
set ANTHROPIC_API_KEY=your-key
```

#### 2. Docker Secrets (Docker Swarm)
```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    image: text-to-statistical-test
    secrets:
      - openai_api_key
    environment:
      - OPENAI_API_KEY_FILE=/run/secrets/openai_api_key

secrets:
  openai_api_key:
    file: ./secrets/openai_api_key.txt
```

#### 3. 클라우드 시크릿 관리
- **AWS**: Systems Manager Parameter Store, Secrets Manager
- **GCP**: Secret Manager
- **Azure**: Key Vault

### GitHub 저장소 공개 시 체크리스트

- [ ] `.env` 파일이 `.gitignore`에 포함되어 있는지 확인
- [ ] `env.example`에는 실제 키가 아닌 예시 텍스트만 있는지 확인
- [ ] 코드 내에 하드코딩된 API 키가 없는지 검사
- [ ] README.md에 환경변수 설정 방법이 명시되어 있는지 확인

## 8단계 워크플로우

1. **데이터 파일 선택**: `input_data/data_files/` 디렉토리의 데이터 파일 선택
2. **사용자 요청 분석**: 자연어로 분석 목표 정의
3. **데이터 요약**: 기술 통계 및 데이터 특성 분석
4. **분석 방법 제안**: AI가 최적의 통계 방법 제안
5. **사용자 선택**: 제안된 방법 중 선택 또는 수정
6. **Agent 분석 계획**: 상세한 분석 실행 계획 수립
7. **Agent 통계 실행**: 자율적 통계 검정 수행
8. **Agent 보고서 생성**: 종합 분석 보고서 작성

## 지원하는 통계 분석

### 그룹 비교 분석
- 독립표본 t-검정
- 대응표본 t-검정
- 일원분산분석 (ANOVA)
- 이원분산분석 (Two-way ANOVA)
- Mann-Whitney U 검정
- Kruskal-Wallis 검정

### 관계 및 상관관계 분석
- 피어슨 상관분석
- 스피어만 상관분석
- 단순/다중 선형회귀분석
- 로지스틱 회귀분석

### 범주형 데이터 분석
- 카이제곱 독립성 검정
- Fisher의 정확검정
- McNemar 검정

### 전제조건 검증 및 사후 분석
- 정규성 검정 (Shapiro-Wilk, Kolmogorov-Smirnov)
- 등분산성 검정 (Levene's, Bartlett's)
- 사후 검정 (Tukey HSD, Bonferroni)
- 효과 크기 계산 (Cohen's d, Eta-squared)

## 프로젝트 구조

```
text-to-statistical-test/
├── main.py                  # 메인 진입점
├── core/                    # 핵심 로직
│   ├── pipeline/            # 8단계 워크플로우
│   ├── rag/                 # RAG 시스템
│   ├── agent/               # Agentic AI
│   ├── workflow/            # 워크플로우 관리
│   └── reporting/           # 보고서 생성
├── services/                # 외부 서비스 연동
│   ├── llm/                 # LLM 서비스
│   ├── statistics/          # 통계 분석
│   ├── code_executor/       # 코드 실행
│   └── visualization/       # 시각화
├── utils/                   # 유틸리티
├── input_data/              # 입력 데이터
├── output_data/             # 출력 결과
├── resources/               # 지식 베이스
└── logs/                    # 로그 파일
```

## 환경변수 설정

### 필수 설정 (최소 하나는 필요)

```bash
# OpenAI API 키
OPENAI_API_KEY=sk-your-actual-openai-api-key-here

# Anthropic API 키 (Claude)
ANTHROPIC_API_KEY=your-actual-anthropic-api-key-here
```

### 애플리케이션 설정

```bash
# 디버그 모드
DEBUG=false

# 로그 레벨 (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# 실행 환경
ENVIRONMENT=production
```

### RAG 및 LLM 설정

```bash
# 기본 LLM 제공자 (openai 또는 anthropic)
DEFAULT_LLM_PROVIDER=openai

# 모델 설정
OPENAI_MODEL=gpt-4-turbo-preview
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# 임베딩 모델
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# 벡터 저장소 유형
VECTOR_STORE_TYPE=faiss
```

### 성능 및 보안 설정

```bash
# 캐시 설정
CACHE_TTL_SECONDS=3600

# 메모리 제한
MAX_MEMORY_MB=2048

# 병렬 처리
PARALLEL_WORKERS=4

# 안전 코드 실행
SAFE_CODE_EXECUTION=true

# 샌드박스 타임아웃
SANDBOX_TIMEOUT=30

# 최대 파일 크기
MAX_FILE_SIZE_MB=100
```

## 문제 해결

### 일반적인 문제들

#### 1. API 키 오류
```
❌ 다음 필수 환경 변수가 설정되지 않았습니다: OPENAI_API_KEY
```

**해결 방법:**
1. `.env` 파일이 존재하는지 확인
2. API 키가 올바르게 설정되었는지 확인
3. `.env` 파일에 공백이나 따옴표가 없는지 확인

#### 2. Docker 환경변수 문제
```
⚠️ .env 파일을 찾을 수 없습니다.
```

**해결 방법:**
```bash
# 환경변수 파일이 있는지 확인
ls -la .env

# Docker에서 환경변수 파일 사용
docker run --env-file .env text-to-statistical-test

# 또는 개별 환경변수 설정
docker run -e OPENAI_API_KEY="your-key" text-to-statistical-test
```

#### 3. 권한 오류 (Unix 시스템)
```bash
# setup.sh 실행 권한 부여
chmod +x setup.sh

# 로그 디렉토리 권한 문제
sudo chown -R $USER:$USER logs/
```

### 로그 확인

```bash
# 애플리케이션 로그 확인
tail -f logs/app.log

# Docker 로그 확인
docker logs text-to-statistical-test
```

## 개발자 가이드

### 개발 환경 설정

```bash
# 개발 의존성 포함 설치
poetry install --with dev

# 코드 품질 검사
poetry run black .
poetry run flake8 .
poetry run mypy .

# 테스트 실행
poetry run pytest
```

### 기여 방법

1. Fork 저장소
2. 기능 브랜치 생성 (`git checkout -b feature/새기능`)
3. 변경 사항 커밋 (`git commit -am '새 기능 추가'`)
4. 브랜치에 Push (`git push origin feature/새기능`)
5. Pull Request 생성

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 지원

문제가 발생하거나 질문이 있으시면:
- GitHub Issues에 문제 보고
- 문서 검토: 위의 문제 해결 섹션 참조
- 환경변수 설정 재확인

---

**주의**: 이 시스템은 API 키를 사용하므로 사용량에 따른 비용이 발생할 수 있습니다. API 사용량을 모니터링하시기 바랍니다. 