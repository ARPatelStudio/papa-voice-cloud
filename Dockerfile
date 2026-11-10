FROM python:3.12-slim

# Install system utilities for downloading & extracting Piper Linux C++ engine
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    tar \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Download & extract official Piper Linux x86_64 C++ ONNX engine
RUN mkdir -p /app/piper_engine && \
    curl -L "https://github.com/rhasspy/piper/releases/latest/download/piper_linux_x86_64.tar.gz" \
    | tar -xz -C /app/piper_engine && \
    chmod +x /app/piper_engine/piper/piper

# Install FastAPI dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy all repository files into container
COPY . /app

ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
