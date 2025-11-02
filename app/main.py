# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFile
import numpy as np, io, os, time, threading

# --------- OPTIONAL: enable S3 download if you keep the model in S3 ----------
USE_S3 = os.getenv("USE_S3", "0") == "1"
if USE_S3:
    import boto3

# --------- RF-DETR model import (giống code train/infer của bạn) -------------
from rfdetr import RFDETRNano

# --------- Config qua ENV -----------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "/models/checkpoint_best_total.pth")
S3_BUCKET  = os.getenv("S3_BUCKET", "")
S3_KEY     = os.getenv("MODEL_KEY", "checkpoint_best_total.pth")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
SCORE_THR  = float(os.getenv("THR", "0.5"))
MAX_PIXELS = int(os.getenv("MAX_PIXELS", "20000000"))  # 20MP safety
ALLOW_ORIGINS = os.getenv("CORS_ORIGINS", "*")

# --------- Class mapping (GIỮ NHẤT QUÁN) -------------------------------------
CLASSES = {0: "with_mask", 1: "without_mask", 2: "mask_weared_incorrect"}

# --------- PIL safety for truncated images -----------------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = FastAPI(title="RF-DETR Face Mask Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOW_ORIGINS] if ALLOW_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_model_lock = threading.Lock()

def _download_model_from_s3():
    if not USE_S3:
        return
    if not S3_BUCKET or not S3_KEY:
        raise RuntimeError("S3 download enabled but S3_BUCKET/MODEL_KEY not set")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.download_file(S3_BUCKET, S3_KEY, MODEL_PATH)
    print(f"✅ Downloaded model from s3://{S3_BUCKET}/{S3_KEY} -> {MODEL_PATH}")

def _load_model_once():
    global _model
    if _model is not None:
        return
    with _model_lock:
        if _model is not None:
            return
        if USE_S3 and not os.path.exists(MODEL_PATH):
            _download_model_from_s3()
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"MODEL_PATH not found: {MODEL_PATH}")
        t0 = time.time()
        m = RFDETRNano(pretrain_weights=MODEL_PATH)
        m.eval()  # very important for inference determinism
        _model = m
        print(f"✅ RF-DETR loaded in {time.time()-t0:.2f}s: {MODEL_PATH}")
        print("✅ CLASS MAPPING:", CLASSES)
        print("✅ THRESHOLD:", SCORE_THR)

@app.on_event("startup")
def _startup():
    _load_model_once()

@app.get("/")
def root():
    return {
        "name": "RF-DETR Face Mask Detection API",
        "version": "1.0.0",
        "health": "/healthz",
        "predict": "/predict",
        "classes": CLASSES,
        "threshold": SCORE_THR,
    }

@app.get("/healthz")
def healthz():
    try:
        _load_model_once()
        return {"status": "ok"}
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Basic MIME guard
    if file.content_type not in {"image/jpeg", "image/png", "image/webp", "image/bmp"}:
        raise HTTPException(status_code=400, detail=f"Unsupported content-type: {file.content_type}")

    try:
        _load_model_once()
        raw = await file.read()
        if len(raw) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        # Safety: limit pixels to avoid huge images DoS
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        if img.width * img.height > MAX_PIXELS:
            # downscale while preserving aspect ratio
            scale = (MAX_PIXELS / (img.width * img.height)) ** 0.5
            new_w, new_h = max(1, int(img.width * scale)), max(1, int(img.height * scale))
            img = img.resize((new_w, new_h))

        arr = np.array(img)  # RGB ndarray (H,W,3)
        t0 = time.time()
        # no_grad for speed + memory
        import torch
        with torch.no_grad():
            dets = _model.predict(arr, threshold=SCORE_THR)

        out = []
        # These attributes should exist per your RF-DETR wrappers
        xyxy = getattr(dets, "xyxy", [])
        clsid = getattr(dets, "class_id", [])
        confs = getattr(dets, "confidence", [])
        for (x1, y1, x2, y2), cid, conf in zip(xyxy, clsid, confs):
            cid = int(cid)
            out.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "class_id": cid,
                "label": CLASSES.get(cid, str(cid)),
                "score": float(conf),
            })
        latency_ms = int((time.time() - t0) * 1000)
        return {"detections": out, "latency_ms": latency_ms, "shape": [int(img.height), int(img.width), 3]}
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

