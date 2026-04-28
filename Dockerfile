FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_ENV=docker
ENV APP_HOST=0.0.0.0
ENV APP_PORT=8000
ENV DATA_DIR=/app/data
ENV RAW_DATA_DIR=/app/data/raw
ENV INTERIM_DATA_DIR=/app/data/interim
ENV PROCESSED_DATA_DIR=/app/data/processed
ENV VECTOR_STORE_DIR=/app/data/vector_store
ENV EVALUATION_DATA_DIR=/app/data/evaluation

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY ui ./ui
COPY data ./data
COPY main.py README.md pyproject.toml ./

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
