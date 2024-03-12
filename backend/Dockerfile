# Backend Dockerfile
FROM python:3.11

# Install system dependencies
RUN apt-get update && apt-get install -y libmagic1 && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Set the working directory
WORKDIR /backend

# Copy the rest of application code
COPY . .

# Install dependencies
# --no-dev: Skip installing packages listed in the [tool.poetry.dev-dependencies] section
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-dev

RUN playwright install-deps
RUN playwright install

RUN mv google_cloud_application_credentials.json /root/.google_cloud_application_credentials.json

ENTRYPOINT [ "uvicorn", "app.server:app", "--host", "0.0.0.0" ]
