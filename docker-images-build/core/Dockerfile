FROM python:3.12
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

ARG PYTHON_VERSION=python3.12
ARG PIP_VERSION=23.3.2


# OS packages
RUN apt-get update --fix-missing; exit 0
RUN apt-get install -y --no-install-recommends \
    curl \
    wget \
    git-core \
    htop \
    unzip \
    zsh \
    vim \
    parallel \
    gcc \
    python3-setuptools \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/tmp/* /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python && \
  cd /usr/local/bin && \
  ln -s /opt/poetry/bin/poetry && \
  poetry config virtualenvs.create false


WORKDIR /tmp


# Install llm_tools 
COPY ./llm_tools  /tmp/llm_tools 
COPY ./pyproject.toml /tmp/pyproject.toml

RUN --mount=type=cache,target=/root/.cache \
    poetry lock && poetry install --no-dev
# Clean
RUN rm -rf /tmp/*

WORKDIR /root

