FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libopenmpi-dev \
        openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        mlflow \
        mpi4py \
        numpy \
        Pillow \
        PyYAML \
        torch \
        torchvision \
        transformers

WORKDIR /workspace/Split-Framework

CMD ["python", "setup/main.py", "--help"]