FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --timeout 120 --retries 5 -r requirements.txt && python -m pip check
COPY . .
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --create-home app && mkdir -p /app/logs /mlflow/artifacts && chown -R app:app /app /mlflow
USER app
RUN python app.py --check
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
