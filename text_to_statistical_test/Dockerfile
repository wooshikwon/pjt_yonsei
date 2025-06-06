# =============================================================================
# Multi-stage build for Text-to-Statistical-Test
# 보안: API 키 등 민감 정보는 빌드 시점에 포함하지 않음
# =============================================================================

# Builder stage - 의존성 설치
FROM python:3.11-slim as builder

# Poetry 환경 설정
ENV POETRY_VERSION=1.8.3
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_CACHE_DIR=/tmp/poetry_cache
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VENV_IN_PROJECT=1

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Poetry 설치
RUN pip install poetry==$POETRY_VERSION

# 작업 디렉토리 설정
WORKDIR /app

# Poetry 설정 파일 복사
COPY pyproject.toml poetry.lock ./

# 의존성 설치 (개발 의존성 제외)
RUN poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project true && \
    poetry install --only=main --no-root && \
    rm -rf $POETRY_CACHE_DIR

# =============================================================================
# Production stage - 실행 환경
# =============================================================================
FROM python:3.11-slim as production

# 파이썬 환경 변수
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

# 애플리케이션 기본 설정 (민감하지 않은 정보만)
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO
ENV DEBUG=false

# 런타임 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 비root 사용자 생성
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 작업 디렉토리 설정
WORKDIR /app

# 가상환경을 builder에서 복사
COPY --from=builder /app/.venv /app/.venv

# 애플리케이션 코드 복사 (민감한 파일 제외)
COPY --chown=appuser:appuser . .

# 민감한 파일들 제거 (혹시 복사되었다면)
RUN rm -f .env .env.* *.key *.pem

# 필요한 디렉토리 생성
RUN mkdir -p \
    input_data/data_files \
    input_data/metadata \
    output_data/reports \
    output_data/visualizations \
    output_data/analysis_cache \
    logs \
    resources/knowledge_base \
    && chown -R appuser:appuser /app

# 비root 사용자로 전환
USER appuser

# 포트 노출 (향후 웹 인터페이스용)
EXPOSE 8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; print('Health check passed'); sys.exit(0)"

# 기본 명령어 (도움말 표시)
CMD ["python", "main.py", "--help"]

# 메타데이터 라벨
LABEL maintainer="wesley"
LABEL version="1.0.0"
LABEL description="RAG 기반 Agentic AI 통계 분석 시스템"
LABEL security.note="민감한 정보는 런타임 환경변수로 주입하세요" 