FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y python3.11 python3.11-venv python3-pip git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY engine ./engine
COPY kernels ./kernels
COPY api ./api
COPY telemetry ./telemetry
COPY benchmarks ./benchmarks
RUN python3.11 -m pip install --upgrade pip && python3.11 -m pip install .
EXPOSE 8000
CMD ["python3.11", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
