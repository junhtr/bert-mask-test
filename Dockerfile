FROM pytorch/pytorch:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        git \
        less \
        vim-nox \
        tig \
        wget \
        zsh \
        && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt update && \
    apt install -y --no-install-recommends \
        libgl1-mesa-dev \
        libglib2.0-0 \
        && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /
RUN python3 -m pip install -v -r /requirements.txt
