FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Install PyTorch first (cached layer)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install spaCy model
RUN python -m spacy download en_core_web_lg

# Copy project
COPY . .
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python"]
