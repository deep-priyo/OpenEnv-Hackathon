FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# 🔥 IMPORTANT: Ensure Python can resolve 'backend' as a module
ENV PYTHONPATH=/app

# Environment variables
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV EVAL_MODE=true
ENV OPENAI_API_KEY=""
ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=gpt-4o-mini
ENV HF_TOKEN=""

# Expose port
EXPOSE 7860

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# 🔥 CRITICAL FIX: Run as module (NOT as file)
CMD ["python", "-m", "backend.app"]