FROM python:3.10-slim

WORKDIR /app

COPY backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .
COPY ml/saved_models/ ./ml/saved_models/

CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:10000"]