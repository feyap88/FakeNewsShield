import torch
import torch.nn.functional as F
from PIL import Image
from src.model import FakeNewsShield
from src.data_preprocessing import MultimodalFakeNewsDataset

# ================= GLOBAL LOAD (ONCE) =================
device = torch.device("cpu")
model = None
ds = None

try:
    print("🚀 Loading trained FakeNewsShield model...")
    model = FakeNewsShield().to(device)
    model.load_state_dict(torch.load("models/fakenewsshield_model.pth", map_location=device))
    model.eval()
    
    ds = MultimodalFakeNewsDataset()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")

def predict_post(text: str, image):
    if model is None or ds is None:
        return "ERROR", 0.0, "Model not loaded"
    
    try:
        # Text
        inputs = ds.tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        
        # Image
        pixel_values = ds.image_processor(images=image, return_tensors="pt").pixel_values

        with torch.no_grad():
            outputs = model(inputs.input_ids.to(device), pixel_values.to(device))
            probs = F.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        label = "FAKE" if pred == 1 else "REAL"
        return label, round(confidence * 100, 1), None

    except Exception as e:
        return "ERROR", 0.0, str(e)