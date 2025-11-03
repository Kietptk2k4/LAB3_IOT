# Dockerfile (đặt ở repo root)
FROM python:3.11-slim

# ENV PYTHONUNBUFFERED=1 \
#     PORT=8080 \
#     MODEL_PATH=/models/checkpoint_best_total.pth \
#     THR=0.5

WORKDIR /app

# --- THÊM DÒNG NÀY ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*
# ---------------------

# Cài deps
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy mã nguồn + model
COPY app/main.py ./main.py
# COPY checkpoint_best_total.pth /models/checkpoint_best_total.pth

# # ENV cho app
ENV MODEL_PATH=/models/checkpoint_best_total.pth
ENV THR=0.5
ENV PORT=8080

EXPOSE 8080

# Chạy FastAPI
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8080"]
