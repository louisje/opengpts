# Backend Dockerfile
FROM python:3.11

ARG TARGETOS
ARG TARGETARCH
ARG TARGETVARIANT

# Install system dependencies
RUN apt-get update && rm -rf /var/lib/apt/lists/*
RUN wget -O golang-migrate.deb https://github.com/golang-migrate/migrate/releases/download/v4.17.0/migrate.${TARGETOS}-${TARGETARCH}${TARGETVARIANT}.deb \
    && dpkg -i golang-migrate.deb \
    && rm golang-migrate.deb

# Install Poetry
RUN pip install poetry

# Set the working directory
WORKDIR /backend

# Copy the rest of application code
COPY . .

# Install all dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

RUN playwright install-deps
RUN playwright install

RUN mv google_cloud_application_credentials.json /root/.google_cloud_application_credentials.json

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD [ "curl", "-f", "http://localhost:8000/health" ]

ENTRYPOINT [ "uvicorn", "app.server:app", "--host", "0.0.0.0", "--log-config", "log_config.json" ]
