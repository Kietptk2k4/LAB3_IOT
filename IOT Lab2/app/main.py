from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch, torchvision.transforms as T
from PIL import Image
import io, boto3, os

app = FastAPI()

S3_BUCKET = "mask-model-kiet"
MODEL_KEY = "checkpoint_best_total.pth"
MODEL_PATH = "/models/checkpoint_best_total.pth"

def download_model_from_s3():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("/models", exist_ok=True)
        s3 = boto3.client("s3", region_name="ap-southeast-1")
        s3.download_file(S3_BUCKET, MODEL_KEY, MODEL_PATH)
        print("✅ Model downloaded from S3")

@app.on_event("startup")
def load_model():
    download_model_from_s3()
    global model, transform, class_names
    model = torch.load(MODEL_PATH, map_location="cpu")
    model.eval()
    class_names = ["with_mask", "no_mask", "mask_incorrect"]
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    print("✅ Model loaded and ready")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            preds = model(img_tensor)
            if isinstance(preds, (list, tuple)):  # in case model returns multi-output
                preds = preds[0]
            probs = torch.nn.functional.softmax(preds, dim=1)
            label_idx = int(torch.argmax(probs))
            score = float(probs[0][label_idx])
            label = class_names[label_idx]
        return JSONResponse({"label": label, "score": round(score, 3)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/healthz")
def health_check():
    return {"status": "ok"}
