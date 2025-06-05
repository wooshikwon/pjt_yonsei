# Python 3.11을 기반으로 한 공식 이미지 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치 (데이터 처리 및 컴파일에 필요한 패키지들)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc-dev \
    libffi-dev \
    libssl-dev \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 파일 복사
COPY requirements.txt .

# Python 패키지 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 로그 디렉토리 생성
RUN mkdir -p logs

# 환경변수 설정
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 기본 환경변수 (실제 사용시 .env 파일이나 docker run 시 설정)
ENV LLM_PROVIDER=openai
ENV LLM_MODEL_NAME=gpt-4o
ENV LOG_LEVEL=INFO
ENV DEBUG_MODE=false

# 포트 노출 (필요시)
EXPOSE 8000

# 헬스체크 스크립트 생성 (데이터 처리 라이브러리 확인)
RUN echo '#!/bin/bash\npython -c "import pandas, numpy, scipy, sklearn, matplotlib, pyarrow; print(\"Dependencies OK\")"' > /app/healthcheck.sh && \
    chmod +x /app/healthcheck.sh

# 헬스체크 설정
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /app/healthcheck.sh

# 기본 명령어 설정
CMD ["python", "main_runner.py", "--help"] 