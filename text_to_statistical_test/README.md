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
python -m src.main --file "team_sales_performance.csv" --request "팀별 영업 성과에 차이가 있는지 분석해줘"
```

#### Docker 환경에서:

```bash
docker-compose run --rm statistical-analyzer --file "team_sales_performance.csv" --request "팀별 영업 성과에 차이가 있는지 분석해줘"
```

## 📁 프로젝트 구조

```
text_to_statistical_test/
├── 📄 README.md                    # 이 파일
├── 📄 BLUEPRINT.md                 # 상세 설계 문서
├── 📄 pyproject.toml               # Poetry 의존성 관리
├── 📄 docker-compose.yml           # Docker 설정
├── 📄 .env                         # 환경변수 (생성 필요)
├── 📂 input_data/
│   └── 📂 data_files/              # 분석할 데이터 파일들
│       ├── team_sales_performance.csv
│       ├── customer_survey.csv
│       └── ... (7개 샘플 파일)
├── 📂 output_data/
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
python -m src.main --file "<데이터파일명>" --request "<자연어 요청>"
```

### 실제 사용 예시

```bash
# 1. t-검정 예시
python -m src.main --file "team_sales_performance.csv" --request "A팀과 B팀의 매출에 유의미한 차이가 있나요?"

# 2. ANOVA 예시  
python -m src.main --file "marketing_campaign_analysis.csv" --request "마케팅 캠페인 유형별로 전환율에 차이가 있는지 분석해주세요"

# 3. 회귀분석 예시
python -m src.main --file "house_price_prediction.csv" --request "집 크기와 가격 사이의 관계를 분석하고 예측 모델을 만들어주세요"

# 4. 상관분석 예시
python -m src.main --file "employee_performance_correlation.csv" --request "직원 만족도와 성과 간의 상관관계를 분석해주세요"
```

### 🔧 고급 설정

#### RAG 시스템 제어

```bash
# RAG 없이 분석 (빠른 실행)
# .env에서 USE_RAG=False로 설정

# 지식 베이스 업데이트 후 벡터 재구축
# .env에서 REBUILD_VECTOR_STORE=True로 설정하고 실행
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

## 📊 샘플 데이터

시스템에는 다양한 분석 시나리오를 테스트할 수 있는 7개의 샘플 데이터셋이 포함되어 있습니다:

| 파일명 | 분석 유형 | 설명 |
|--------|-----------|------|
| `team_sales_performance.csv` | t-검정, ANOVA | 팀별 영업 성과 데이터 |
| `marketing_campaign_analysis.csv` | ANOVA | 마케팅 캠페인 효과 분석 |
| `house_price_prediction.csv` | 선형회귀 | 주택 가격 예측 모델링 |
| `student_admission_data.csv` | 로지스틱회귀 | 대학 입학 예측 분석 |
| `employee_performance_correlation.csv` | 상관분석 | 직원 성과 요인 분석 |
| `manufacturing_quality_control.csv` | Z-검정 | 제조업 품질 관리 |
| `customer_survey.csv` | 카이제곱 검정 | 고객 설문 분석 |

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
