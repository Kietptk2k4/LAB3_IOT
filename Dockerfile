# Dockerfile (đặt ở repo root)
FROM python:3.11-slim

WORKDIR /app

# Cài deps
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy mã nguồn + model
COPY /app/main.py ./main.py
COPY checkpoint_best_total.pth /models/checkpoint_best_total.pth

# ENV cho app
ENV MODEL_PATH=/models/checkpoint_best_total.pth
ENV THR=0.5
ENV PORT=8080

EXPOSE 8080

# Chạy FastAPI
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8080"]
