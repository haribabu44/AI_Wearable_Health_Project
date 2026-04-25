FROM python:3.11

WORKDIR /app

COPY backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

CMD ["gunicorn", "app.main:app", "--bind", "0.0.0.0:10000"]