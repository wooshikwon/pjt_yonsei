#!/usr/bin/env python3
"""
프로젝트 초기 설정 스크립트

이 스크립트는 text_to_statistical_test 프로젝트의 초기 설정을 수행합니다.
자연어 요청 기반 AI 추천 통계 분석 도구를 위한 환경을 구성합니다.

- 필요한 디렉토리 생성
- 기본 설정 파일 생성  
- 샘플 데이터 설정
- Poetry 기반 의존성 검증
- Docker 배포 준비
- 초기 검증
"""

import os
import sys
from pathlib import Path
import logging

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def create_directories():
    """필요한 디렉토리들을 생성합니다"""
    logger = logging.getLogger(__name__)
    
    directories = [
        # 데이터 관련 디렉토리
        'input_data',
        'output_results',
        
        # 리소스 디렉토리
        'resources/rag_index',
        'resources/code_snippets/descriptive_stats',
        'resources/code_snippets/t_test',
        'resources/code_snippets/anova',
        'resources/code_snippets/chi_square',
        'resources/code_snippets/correlation',
        'resources/code_snippets/regression',
        'resources/code_snippets/normality_tests',
        
        # LLM 서비스 프롬프트 디렉토리
        'llm_services/prompts/common',
        'llm_services/prompts/request_analysis',
        'llm_services/prompts/method_recommendation',
        'llm_services/prompts/assumption_checking',
        'llm_services/prompts/result_interpretation',
        
        # 로그 디렉토리
        'logs'
    ]
    
    for directory in directories:
        path = Path(directory)
        try:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"디렉토리 생성 완료: {directory}")
        except Exception as e:
            logger.error(f"디렉토리 생성 실패: {directory} - {e}")
            return False
    
    return True

def create_env_file():
    """환경 설정 파일을 생성합니다"""
    logger = logging.getLogger(__name__)
    
    env_content = """# LLM 서비스 설정 (OpenAI만 사용)
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-4o

# OpenAI API 키 (실제 키로 교체해야 함)
OPENAI_API_KEY=sk-proj-your_actual_openai_api_key_here

# 임베딩 모델 설정
EMBEDDING_MODEL_NAME=text-embedding-ada-002

# 컨텍스트 관리 설정
MAX_HISTORY_ITEMS=20
CONTEXT_TOKEN_LIMIT=3000

# 코드 실행 설정
CODE_EXECUTION_TIMEOUT=30
SAFE_CODE_EXECUTION=true

# 보고서 설정
REPORT_FORMAT=html

# 로깅 설정
LOG_LEVEL=INFO

# 개발 모드
DEBUG_MODE=false

# 자연어 요청 분석 설정
RECOMMENDATION_COUNT_MAX=3
CONFIDENCE_THRESHOLD=0.4
AUTO_ASSUMPTION_CHECK=true
"""
    
    try:
        if not Path('.env').exists():
            with open('.env', 'w', encoding='utf-8') as f:
                f.write(env_content)
            logger.info(".env 파일 생성 완료")
        else:
            logger.info(".env 파일이 이미 존재합니다")
        return True
    except Exception as e:
        logger.error(f".env 파일 생성 실패: {e}")
        return False

def create_sample_data_info():
    """input_data 폴더에 샘플 데이터 정보를 생성합니다"""
    logger = logging.getLogger(__name__)
    
    readme_content = """# Input Data 폴더

이 폴더에 분석할 데이터 파일들을 저장하세요.

## 지원하는 파일 형식

- **CSV** (`.csv`) - 가장 권장되는 형식
- **Excel** (`.xlsx`, `.xls`) - 첫 번째 시트가 자동 로드됨
- **Parquet** (`.parquet`) - 대용량 데이터에 적합
- **JSON** (`.json`) - 중첩된 구조의 데이터

## 샘플 데이터 예시

1. **survey_data.csv** - 설문 조사 데이터
   - 컬럼: age, gender, score, group, satisfaction
   - 용도: 그룹별 평균 비교, 상관관계 분석

2. **experiment_results.xlsx** - 실험 결과 데이터  
   - 컬럼: subject_id, treatment, before_score, after_score
   - 용도: 대응표본 t-검정, 효과 분석

3. **sales_data.csv** - 판매 데이터
   - 컬럼: date, region, product, sales_amount, customer_type
   - 용도: 회귀분석, 분산분석

## 데이터 준비 팁

- 첫 번째 행에 컬럼명 포함
- 한글 컬럼명 사용 가능
- 결측치는 빈 셀 또는 'NA'로 표시
- 날짜는 YYYY-MM-DD 형식 권장
"""
    
    try:
        readme_path = Path("input_data/README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        logger.info("input_data/README.md 파일 생성 완료")
        return True
    except Exception as e:
        logger.error(f"input_data/README.md 파일 생성 실패: {e}")
        return False

def verify_setup():
    """설정이 올바르게 완료되었는지 검증합니다"""
    logger = logging.getLogger(__name__)
    
    required_files = [
        'main.py',
        'core/agent.py',
        'core/workflow_manager.py',
        'core/decision_engine.py',
        'core/context_manager.py',
        'llm_services/llm_client.py',
        'llm_services/prompt_crafter.py',
        'data_processing/data_loader.py',
        'rag_system/code_retriever.py',
        'code_execution/safe_code_executor.py',
        'reporting/report_generator.py',
        'config/settings.py',
        'resources/workflow_graph.json',
        'utils/__init__.py',
        'utils/analysis_recommender.py',
        'utils/workflow_utils.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.warning("다음 파일들이 누락되었습니다:")
        for file_path in missing_files:
            logger.warning(f"  - {file_path}")
        return False
    
    logger.info("모든 필수 파일이 존재합니다")
    return True

def verify_poetry_setup():
    """Poetry 설정 및 의존성 검증"""
    logger = logging.getLogger(__name__)
    
    # pyproject.toml 파일 존재 확인
    if not Path('pyproject.toml').exists():
        logger.error("pyproject.toml 파일이 존재하지 않습니다")
        logger.info("Poetry 프로젝트가 아닌 것 같습니다. 'poetry init'을 실행하세요")
        return False
    
    logger.info("pyproject.toml 파일 확인 완료")
    
    # Poetry가 설치되어 있는지 확인
    try:
        import subprocess
        result = subprocess.run(['poetry', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Poetry 설치 확인: {result.stdout.strip()}")
        else:
            logger.warning("Poetry가 설치되지 않았습니다")
            return False
    except FileNotFoundError:
        logger.warning("Poetry가 시스템에 설치되지 않았습니다")
        logger.info("Poetry 설치: curl -sSL https://install.python-poetry.org | python3 -")
        return False
    
    # 핵심 의존성 확인
    try:
        import pandas
        import numpy  
        import scipy
        logger.info("핵심 데이터 분석 라이브러리 확인 완료")
    except ImportError as e:
        logger.warning(f"일부 라이브러리가 설치되지 않았습니다: {e}")
        logger.info("'poetry install'을 실행하여 의존성을 설치하세요")
        return False
    
    return True

def verify_docker_setup():
    """Docker 관련 파일 확인"""
    logger = logging.getLogger(__name__)
    
    docker_files = ['Dockerfile', 'docker-compose.yml', '.dockerignore']
    existing_docker_files = []
    
    for docker_file in docker_files:
        if Path(docker_file).exists():
            existing_docker_files.append(docker_file)
    
    if existing_docker_files:
        logger.info(f"Docker 관련 파일 확인: {', '.join(existing_docker_files)}")
    else:
        logger.info("Docker 관련 파일이 없습니다 (필요시 추가)")
    
    return True

def main():
    """메인 실행 함수"""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("🔬 Statistical Analysis Assistant 프로젝트 초기 설정")
    logger.info("   자연어 요청 기반 AI 추천 분석 도구 (Poetry + Docker)")
    logger.info("=" * 60)
    
    success = True
    
    # 1. Poetry 설정 확인
    logger.info("1. Poetry 설정 및 의존성을 확인합니다...")
    if not verify_poetry_setup():
        logger.warning("Poetry 설정에 문제가 있지만 계속 진행합니다")
    
    # 2. 디렉토리 생성
    logger.info("2. 필요한 디렉토리들을 생성합니다...")
    if not create_directories():
        success = False
    
    # 3. 환경 설정 파일 생성
    logger.info("3. 환경 설정 파일을 생성합니다...")
    if not create_env_file():
        success = False
    
    # 4. 샘플 데이터 정보 생성
    logger.info("4. input_data 폴더 정보를 생성합니다...")
    if not create_sample_data_info():
        success = False
    
    # 5. 설정 검증
    logger.info("5. 프로젝트 설정을 검증합니다...")
    if not verify_setup():
        logger.warning("일부 파일이 누락되었지만 기본 설정은 완료되었습니다")
    
    # 6. Docker 설정 확인
    logger.info("6. Docker 관련 설정을 확인합니다...")
    verify_docker_setup()
    
    logger.info("=" * 60)
    if success:
        logger.info("✅ 프로젝트 초기 설정이 성공적으로 완료되었습니다!")
        logger.info("")
        logger.info("📋 다음 단계:")
        logger.info("1. .env 파일에서 API 키를 실제 값으로 교체하세요")
        logger.info("   - OPENAI_API_KEY")
        logger.info("2. Poetry 의존성 설치: poetry install")
        logger.info("3. 분석할 데이터를 input_data 폴더에 저장하세요")
        logger.info("4. 실행: poetry run python main.py")
        logger.info("")
        logger.info("🐳 Docker 실행 (선택사항):")
        logger.info("   - Docker 이미지 빌드: docker build -t statistical-assistant .")
        logger.info("   - Docker 컨테이너 실행: docker run -it statistical-assistant")
        logger.info("")
        logger.info("🗣️ 자연어 요청 예시:")
        logger.info("  • '그룹별로 점수 평균에 차이가 있는지 알고 싶어요'")
        logger.info("  • '두 변수 간에 관계가 있나요?'")
        logger.info("  • '성별과 선호도에 관련이 있나요?'")
    else:
        logger.error("❌ 프로젝트 초기 설정 중 일부 오류가 발생했습니다")
        sys.exit(1)
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main() 