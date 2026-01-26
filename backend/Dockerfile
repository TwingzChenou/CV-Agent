# Builder stage
FROM python:3.11-slim as builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create a non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port (Render sets PORT env var, but we expose 8000 as default/doc)
EXPOSE 8000

# Start command
# Using shell form to properly expand $PORT if needed, but uvicorn handles it via our main.py or args.
# Here we use the direct uvicorn command as typically recommended for containers.
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
# However, for Render, we want to respect the PORT env var.
# We can do this by using sh -c
CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"
