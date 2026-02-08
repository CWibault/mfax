FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
LABEL org.opencontainers.image.authors=""

ENV CUDA_PATH=/usr/individual/cuda
ENV CUDA_INCLUDE_PATH=/usr/individual/cuda/include
ENV CUDA_LIBRARY_PATH=/usr/individual/cuda/lib64

# --- timezone ---
ENV TZ=Europe/London DEBIAN_FRONTEND=noninteractive

# --- add old libraries (Python 3.10) to Ubuntu 22.04 ---
RUN apt-get update && apt-get install -y software-properties-common && \
    apt-add-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    patchelf \
    libglew-dev \
    git \
    curl \
    python3.10 \
    python3-pip \
    python3.10-venv \
    python3-setuptools \
    python3-wheel \
    python3.10-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --- create individual user ---
ARG UID
ARG GID
RUN if [ ${UID:-0} -ne 0 ] && [ ${GID:-0} -ne 0 ]; then \
    groupadd -g ${GID} duser && \
    useradd -l -u ${UID} -g duser duser && \
    install -d -m 0755 -o duser -g duser /home/duser && \
    chown --changes --silent --no-dereference --recursive ${UID}:${GID} /home/duser; \
    fi

USER duser
WORKDIR /home/duser

# --- install Python packages ---
ENV PATH="/home/duser/.individual/bin:$PATH"
RUN python3.10 -m pip install --upgrade pip
ARG REQS
RUN python3.10 -m pip install --default-timeout=100 $REQS -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

WORKDIR /home/duser/mfax