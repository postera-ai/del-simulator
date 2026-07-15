# syntax=docker/dockerfile:1.4
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

WORKDIR /app/del-simulator

# Set the environment variable to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Must be set before the Poetry installer runs below -- it reads these env vars at install
# time to decide the version to fetch and where to put it.
ENV POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH:/root/.local/bin"

# Install Python 3.11, tmux, and other dependencies
RUN apt update && apt install -y curl software-properties-common tmux \
    && add-apt-repository -y 'ppa:deadsnakes/ppa' \
    && apt update && apt install -y python3.11 python3.11-venv python3.11-dev \
    && curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.11 get-pip.py \
    && rm get-pip.py \
    && python3.11 -m pip install poethepoet==0.24.1 \
    && ln -s /usr/bin/python3.11 /usr/bin/python \
    && curl -sSL https://install.python-poetry.org | python3.11 - \
    && ln -s "$POETRY_HOME/bin/poetry" /usr/local/bin/poetry \
    && rm -rf /var/lib/apt/lists/*

# Copy the pyproject.toml file
COPY pyproject.toml ./
# Install dependencies
RUN poetry install --no-root \
    && rm -rf $(find /app -type d -name __pycache__) \
    && rm -rf /root/.cache/pip /root/.cache/pypoetry/cache /root/.cache/pypoetry/artifacts

# Copy the rest of the application files
COPY . .

# Install the package itself into poetry's virtualenv (not the system python) so it lands
# alongside its dependencies
RUN poetry run pip install --no-deps .
