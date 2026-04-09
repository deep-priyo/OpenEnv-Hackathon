FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# IMPORTANT: /app is the root, so 'backend' is importable as a package
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

# Healthcheck - more forgiving during startup
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=5 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the Flask app directly (simplest and most reliable)
CMD ["python", "backend/app.py"]
