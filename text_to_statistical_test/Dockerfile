# =============================================================================
# Multi-stage Docker build for Text-to-Statistical-Test
# =============================================================================

# 1. Builder Stage: Install dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

# Set up Poetry environment
ENV POETRY_VERSION=1.8.2
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VENV_IN_PROJECT=true
ENV POETRY_NO_INTERACTION=1
ENV PATH="$POETRY_HOME/bin:$PATH"

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Set working directory
WORKDIR /app

# Copy dependency definition files
COPY pyproject.toml poetry.lock ./

# Install only production dependencies
# This creates a .venv directory in /app
RUN poetry install --no-root --no-dev


# 2. Final Stage: Create the production image
# -----------------------------------------------------------------------------
FROM python:3.11-slim as final

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/app/.venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create a non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv ./.venv

# Copy application source code
COPY --chown=appuser:appgroup . .

# Create necessary directories and set permissions
# Ensures the app can write reports and logs
RUN mkdir -p output_data/reports output_data/visualizations logs && \
    chown -R appuser:appgroup output_data logs

# Switch to the non-root user
USER appuser

# Entrypoint to run the application
# Expects a .env file to be mounted or environment variables to be provided at runtime.
# Example: python main.py --file 'input_data/data_files/sample_data.csv' --request 'Compare sales between Group A and Group B'
CMD ["python", "main.py"]

# 메타데이터 라벨
LABEL maintainer="wesley"
LABEL version="1.0.0"
LABEL description="RAG 기반 Agentic AI 통계 분석 시스템"
LABEL security.note="민감한 정보는 런타임 환경변수로 주입하세요"

# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the pyproject.toml and poetry.lock files into the container
# We copy these first to leverage Docker's layer caching.
COPY pyproject.toml poetry.lock* /app/

# Install poetry
RUN pip install poetry

# Install project dependencies
# --no-root: Don't install the project itself, only its dependencies
# This is useful for creating a runtime environment for the sandbox
RUN poetry install --no-root

# The sandbox doesn't need the project source code, only the installed libraries.
# The code to be executed will be mounted dynamically into the container.

# This command is not strictly necessary for the sandbox, but can be useful for debugging.
CMD ["python", "--version"] 