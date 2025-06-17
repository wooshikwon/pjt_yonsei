# 1. Base Image: Use an official, slim Python image
FROM python:3.11-slim

# 2. Environment Variables: For unbuffered logs and Poetry configuration
ENV PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

# 3. Install Poetry
RUN pip install poetry

# 4. Work Directory: Set the working directory inside the container
WORKDIR /app

# 5. Layer Caching: Copy only dependency files first to leverage Docker cache
COPY pyproject.toml poetry.lock ./

# 6. Install Dependencies: Install only production dependencies
RUN poetry install --no-dev --no-root

# 7. Copy Application Code: Copy the source code and necessary resources
COPY src ./src
COPY input_data ./input_data
COPY resources ./resources

# 8. Command: Set the default command to run the application
# The user will provide --file and --request arguments when running the container.
CMD ["poetry", "run", "python", "src/main.py"] 