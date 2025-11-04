# Use Python 3.11 as base image (adjust version as needed)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies that might be needed for torch and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx
ENV PATH="/usr/local/bin:${PATH}"

# Declare build argument for Guardrails token
ARG GUARDRAILS_TOKEN
ENV GUARDRAILS_TOKEN=${GUARDRAILS_TOKEN}

# Copy pyproject.toml and any lock files
COPY pyproject.toml ./
# Copy uv.lock if it exists (optional)
COPY uv.lock* ./

# Install dependencies from pyproject.toml
RUN uv sync


# Configure guardrails with token and install hub packages
RUN uv run guardrails configure --token ${GUARDRAILS_TOKEN} --enable-metrics --enable-remote-inferencing && \
    uv run guardrails hub install hub://tryolabs/restricttotopic && \
    uv run guardrails hub install hub://guardrails/detect_jailbreak && \
    uv run guardrails hub install hub://guardrails/competitor_check && \
    uv run guardrails hub install hub://arize-ai/llm_rag_evaluator && \
    uv run guardrails hub install hub://guardrails/profanity_free && \
    uv run guardrails hub install hub://guardrails/guardrails_pii

# Copy .env file for runtime environment variables (optional)
COPY .env* ./

# Copy the rest of your application
COPY . .

# Default command - adjust based on your needs
CMD ["uv", "run", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]