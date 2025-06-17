# 1. Base Image: Use an official, slim Python image
FROM python:3.11-slim

# 2. Environment Variables: For unbuffered logs and Poetry configuration
ENV PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONPATH=/app

# 3. Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Poetry
RUN pip install poetry==1.7.1

# 5. Work Directory: Set the working directory inside the container
WORKDIR /app

# 6. Layer Caching: Copy only dependency files first to leverage Docker cache
COPY pyproject.toml poetry.lock ./

# 7. Install Dependencies: Install production dependencies
RUN poetry install --only=main --no-root

# 8. Copy Application Code: Copy the source code and necessary resources
COPY src ./src
COPY input_data ./input_data
COPY resources ./resources

# 9. Create necessary directories with proper permissions
RUN mkdir -p /app/output_data/reports \
    && mkdir -p /app/logs \
    && chmod -R 755 /app/output_data \
    && chmod -R 755 /app/logs

# 10. Install the package
RUN poetry install --only-root

# 11. Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# 12. Command: Set the default command to run the application
# Usage: docker run <image> --file <filename> --request "<request>"
ENTRYPOINT ["poetry", "run", "python", "-m", "src.main"] 