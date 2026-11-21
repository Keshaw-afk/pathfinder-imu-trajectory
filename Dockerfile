# Reproducible runtime for the Track 3 submission (src/main.py).
#
#   docker build -t pathfinder .
#   docker run --rm -v "$PWD:/work" pathfinder /work/input.zip -o /work/estimated_trajectory.txt
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/

ENTRYPOINT ["python", "/app/src/main.py"]
