# Text-to-Statistical-Test 📊

자연어로 요청하면 자동으로 통계 분석을 수행해주는 LLM 에이전트 시스템

[![Python](https://img.shields.io/badge/Python-3.11+-3776ab.svg)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-00a67e.svg)](https://openai.com/)
[![Poetry](https://img.shields.io/badge/Poetry-package%20manager-blue.svg)](https://python-poetry.org/)
[![Docker](https://img.shields.io/badge/Docker-containerized-2496ed.svg)](https://www.docker.com/)

## 🎯 프로젝트 개요

**Text-to-Statistical-Test**는 복잡한 통계 지식이나 코딩 능력 없이도 누구나 자연어 질문만으로 데이터 기반의 전문적인 통계 분석을 수행할 수 있는 LLM 에이전트 시스템입니다.

### ✨ 주요 특징

- **🗣️ 자연어 인터페이스**: "A와 B 제품 간의 고객 만족도에 차이가 있는지 비교 분석해줘"
- **🤖 자율적 분석 계획**: 데이터 구조를 파악하고 적절한 통계 검정 방법을 자동 선택
- **🔍 RAG 기반 컨텍스트 강화**: 비즈니스 용어와 데이터 컬럼을 지능적으로 연결
- **🛠️ 자가 수정 능력**: 코드 실행 중 오류 발생 시 자동으로 문제를 해결
- **📋 전문적 보고서**: 주요 발견사항, 결론, 권장사항이 포함된 상세 분석 보고서 생성

### 🎪 지원하는 분석 유형

- **t-검정** (독립표본, 대응표본)
- **ANOVA** (일원분산분석, 이원분산분석)
- **회귀분석** (선형회귀, 로지스틱회귀)
- **비율 검정** (Z-검정)
- **상관분석**
- **카이제곱 검정**

## 🚀 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/wooshikwon/pjt_yonsei.git
cd pjt_yonsei/text_to_statistical_test
```

### 2. 환경 설정

#### Option A: Poetry 사용 (권장)

```bash
# Poetry 설치 (없는 경우)
curl -sSL https://install.python-poetry.org | python3 -

# 의존성 설치
poetry install

# 가상환경 활성화
poetry shell
```

#### Option B: Docker 사용

```bash
# Docker 이미지 빌드
docker-compose build
```

### 3. 환경변수 설정

```bash
# env.example을 복사하여 .env 파일 생성
cp env.example .env

# .env 파일 편집
nano .env
```

`.env` 파일에서 다음 설정을 변경하세요:

```env
# OpenAI API 키 설정 (필수)
OPENAI_API_KEY="sk-your-actual-api-key-here"

# RAG 사용 여부 (선택)
USE_RAG=True

# 벡터 저장소 재구축 여부 (선택)
REBUILD_VECTOR_STORE=False
```

### 4. 첫 번째 분석 실행

#### Poetry 환경에서:

```bash
poetry run python -m src.main --file "teachingratings.csv" --request "성별이 교수 평가 점수에 영향을 주는지 알려주세요."
```

#### Docker 환경에서:

```bash
poetry run python -m src.main --file "pima-indians-diabetes.csv" --request "나이와 체질량지수가 당뇨병 발병에 어떤 영향을 주는지 분석해줘."
```

## 📁 프로젝트 구조

```
text_to_statistical_test/
├── 📄 README.md                    # 이 파일
├── 📄 BLUEPRINT.md                 # 상세 설계 문서
├── 📄 pyproject.toml               # Poetry 의존성 관리
├── 📄 docker-compose.yml           # Docker 설정
├── 📄 .env                         # 환경변수 (생성 필요)
├── 📂 input/
│   └── 📂 data_files/              # 분석할 데이터 파일들
│       ├── team_sales_performance.csv
│       ├── customer_survey.csv
│       └── ... (9개 샘플 파일)
├── 📂 output/
│   ├── 📂 data_files/              # 분석에 사용된 전처리된 파일
│   └── 📂 reports/                 # 생성된 분석 보고서
├── 📂 logs/                        # 시스템 로그 파일
├── 📂 resources/
│   ├── 📂 knowledge_base/          # RAG용 지식 베이스
│   └── 📂 rag_index/               # 생성된 벡터 인덱스
└── 📂 src/                         # 핵심 소스 코드
    ├── main.py                     # 메인 실행 파일
    ├── agent.py                    # LLM 에이전트
    └── components/                 # 핵심 컴포넌트들
```

## 💡 사용법

### 기본 명령어 구조

```bash
Poetry run python -m src.main --file "<데이터파일명>" --request "<자연어 요청>"
```

### 실제 사용 예시

`input_data/data_files/`에 포함된 7개의 샘플 데이터와 함께 아래 예시를 바로 실행해볼 수 있습니다.

```bash
# 예시 1: 성별에 따라 교수 평가 점수에 차이가 있는지 분석 (t-검정 유도)
poetry run python -m src.main --file "teachingratings.csv" --request "성별이 교수 평가 점수에 영향을 주는지 알려주세요."

# 예시 2: 연령대별로 외모 점수가 다른지 분석 (ANOVA 유도)
poetry run python -m src.main --file "teachingratings.csv" --request "교수의 외모 점수가 연령대별로 차이가 나는지 궁금합니다. 분석해주세요."

# 예시 3: 외모 점수와 평가 점수의 관계 분석 (다중 회귀분석 유도)
poetry run python -m src.main --file "ames_housing.csv" --request "집 값에 가장 큰 영향을 주는 요인이 뭘까? 예를 들어 집의 전반적인 상태나, 크기, 그리고 언제 지어졌는지가 집 값과 어떤 관계가 있는지 궁금해."

# 예시 4: 당뇨병 발병 여부 예측 (상관관계 분석 유도)
poetry run python -m src.main --file "ames_housing.csv" --request "우리 동네 집 값을 올리는 가장 확실한 숫자 조건 5가지만 알려줘."
```

### 🔧 고급 설정

#### RAG 시스템 제어

```bash
# RAG 없이 분석 (빠른 실행)
# .env에서 USE_RAG=False로 설정

# 지식 베이스 업데이트 후 벡터 재구축
# .env에서 REBUILD_VECTOR_STORE=True로 설정하고 실행
```

#### RAG 지식 베이스 빌드 (선택 사항)

`resources/knowledge_base/`에 있는 문서(.md)들을 임베딩하여 `resources/rag_index/`에 벡터 인덱스를 생성하거나 업데이트합니다. 지식 베이스 문서를 추가/수정한 경우, 아래 명령어를 실행하여 RAG 시스템에 변경사항을 반영해야 합니다.

```bash
poetry run python src/embedder.py
```

#### 사용자 정의 지식 베이스

`resources/knowledge_base/` 디렉토리에 마크다운 파일을 추가하여 도메인별 용어 정의나 비즈니스 컨텍스트를 제공할 수 있습니다.

```markdown
# 예시: resources/knowledge_base/business_terms.md

## 고객 만족도
- 측정 방법: 1-5점 리커트 척도
- 데이터 컬럼: satisfaction_score
- 해석: 3점 이상을 만족으로 간주
```

## 샘플 데이터

시스템에는 다양한 통계 분석 시나리오를 테스트할 수 있는 7개의 공개 데이터셋이 `input_data/data_files/`에 포함되어 있습니다.

| 파일명 | 주요 변수 | 대표 분석 유형 |
|---|---|---|
| `teachingratings.csv` | 교수 평가 점수, 외모 점수, 성별, 연령대 | t-검정, ANOVA, 상관분석 |
| `pima-indians-diabetes.csv` | 당뇨병 발병 여부, 혈당, BMI | 로지스틱 회귀 |
| `titanic.csv` | 생존 여부, 좌석 등급, 성별, 나이 | 카이제곱 검정, 로지스틱 회귀 |
| `tips.csv` | 팁 금액, 총 식사 금액, 요일, 시간 | 선형 회귀, ANOVA |
| `boston_housing.csv` | 주택 가격, 범죄율, 방 개수 | 선형 회귀, 상관분석 |
| `mpg.csv` | 연비, 마력, 무게, 제조국가 | 선형 회귀, ANOVA |
| `Iris.csv` | 붓꽃 종류, 꽃잎/꽃받침 길이/너비 | ANOVA, 분류 |
| `ames_housing.csv` | 주택 가격, 건축 연도, 생활 면적 | 다중 회귀, 상관분석 |
| `heart_disease_uci.csv` | 환자 나이, 혈압, 콜레스트롤 수치 | 로지스틱 회귀, 카이제곱 검정 |

## 🐞 문제 해결

### 자주 발생하는 문제들

**1. OpenAI API 오류**
```bash
# API 키가 올바른지 확인
echo $OPENAI_API_KEY  # 또는 .env 파일 확인
```

**2. 모듈을 찾을 수 없음**
```bash
# 올바른 실행 방법 사용
python -m src.main  # ✅ 맞음
python src/main.py  # ❌ 틀림
```

**3. RAG 인덱스 문제**
```bash
# .env에서 다음과 같이 설정 후 재실행
REBUILD_VECTOR_STORE=True
```

**4. Docker 권한 문제**
```bash
# Docker 그룹에 사용자 추가
sudo usermod -aG docker $USER
# 로그아웃 후 재로그인
```

### 로그 확인

시스템의 상세 로그는 `logs/` 디렉토리에서 확인할 수 있습니다:

```bash
# 최신 로그 확인
tail -f logs/analysis_$(date +%Y%m%d).log
```

## 🧪 테스트 실행

```bash
# 전체 테스트 실행
poetry run pytest

# 특정 테스트만 실행
poetry run pytest tests/test_agent.py

# 상세 출력과 함께 실행
poetry run pytest -v -s
```

## 🤝 기여하기

1. 이 저장소를 Fork하세요
2. 새로운 기능 브랜치를 생성하세요 (`git checkout -b feature/AmazingFeature`)
3. 변경사항을 커밋하세요 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치에 푸시하세요 (`git push origin feature/AmazingFeature`)
5. Pull Request를 생성하세요

---

**개발자**: wesley  
**프로젝트 페이지**: https://github.com/wooshikwon/pjt_yonsei/tree/main/text_to_statistical_test
