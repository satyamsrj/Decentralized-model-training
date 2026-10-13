FROM python:3.11-slim

LABEL maintainer="p2p-ml-swarm"
LABEL description="P2P ML Swarm Worker Node"

WORKDIR /app

# System dependencies for bitsandbytes and torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY p2p_ml_swarm/ ./p2p_ml_swarm/
COPY pyproject.toml .
COPY README.md .

# Install the package
RUN pip install --no-cache-dir -e .

# Copy config and examples
COPY config/ ./config/
COPY examples/ ./examples/

# Default command: run MNIST worker
ENTRYPOINT ["python", "examples/mnist/train.py"]
CMD ["--config", "config/swarm_config.yaml", "--epochs", "5"]
