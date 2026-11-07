FROM python:3.11-slim AS deps

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 libsdl2-2.0-0 libsdl2-mixer-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV CALM_PYTHON=python \
    CALM_PIP=pip \
    SDL_AUDIODRIVER=dummy \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    PYTHONPATH=/workspace/src

WORKDIR /workspace

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt pyproject.toml ./
RUN grep -v '^torch' requirements.txt > req.txt \
    && pip install --no-cache-dir -r req.txt rust-just \
    && pip install --no-cache-dir $(python -c "import tomllib; print(' '.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['optional-dependencies']['dev']))") \
    && rm req.txt

FROM deps

COPY . .
RUN pip install --no-cache-dir -e ".[dev]" \
    && python -m utils.generate_train_scenarios

RUN useradd --create-home --shell /bin/bash calm \
    && chown -R calm:calm /workspace
USER calm

CMD ["bash"]
