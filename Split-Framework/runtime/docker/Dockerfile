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
        docker-cli \
        docker.io \
        git \
        libopenmpi-dev \
        openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

RUN docker_bin="/usr/bin/docker" \
    && printf '#!/bin/sh\ncontainer="$1"\nshift\nexec %s exec -i "$container" /bin/sh -lc "$*"\n' "$docker_bin" > /usr/local/bin/docker-exec \
    && ln -sf "$docker_bin" /usr/local/bin/docker \
    && chmod +x /usr/local/bin/docker-exec

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
