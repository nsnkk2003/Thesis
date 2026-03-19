# Start with a base image that has Python 3.10 + CUDA 12.1 (GPU support)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Install Python and basic tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (Docker caches this layer if requirements don't change)
COPY requirements.txt .

# Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code into the container
COPY src/ ./src/

# Default command: run training
# Can be overridden in Kubernetes to run evaluate.py instead
CMD ["python", "src/train.py"]
