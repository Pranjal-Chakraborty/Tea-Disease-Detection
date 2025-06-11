import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError
import json
import logging

app = FastAPI(title="Tea Leaf Disease Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model from Hugging Face Hub
model = timm.models.create_model("hf_hub:Luna-Skywalker/convnext-tea-v2", pretrained=True)
model.eval()
model.to(device)

# Load class labels from config
labels_path = hf_hub_download(repo_id="Luna-Skywalker/convnext-tea-v2", filename="config.json")
with open(labels_path, "r") as f:
    config = json.load(f)
class_names = config["label_names"]

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Prediction function
def predict(image_file) -> dict:
    image = Image.open(image_file).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1).squeeze().tolist()

    return {class_names[i]: round(probs[i], 4) for i in range(len(class_names))}

# API route
@app.post("/classify")
async def classify_leaf(image: UploadFile = File(...)):
    try:
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
        result = predict(image.file)
        return JSONResponse(content=jsonable_encoder(result), status_code=200)
    
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Unable to read the image. It might be corrupted or unsupported.")
    
    except Exception as e:
        logging.exception("Unexpected error during prediction.")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
