# Dockerfile for HuggingFace Spaces
# Build: docker build -t code-review-env .
# Run:   docker run -p 7860:7860 -e ANTHROPIC_API_KEY=sk-... code-review-env

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# HuggingFace Spaces runs on port 7860
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV EVAL_MODE=true

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/api/health || exit 1

# Start Flask server
CMD ["python", "backend/app.py"]