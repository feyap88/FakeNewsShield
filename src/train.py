import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model import FakeNewsShield
from src.data_preprocessing import MultimodalFakeNewsDataset
import time

print("🚀 FakeNewsShield Training Started...")
print("=====================================")

device = torch.device("cpu")
print(f"Device: {device}")

# === LOAD DATASET ===
print("📊 Loading dataset (this may take 1-2 minutes)...")
start_time = time.time()

dataset = MultimodalFakeNewsDataset(split="train", max_samples=300)  # Small for testing

print(f"✅ Dataset loaded successfully! ({len(dataset)} samples) - Time: {time.time()-start_time:.1f}s")

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# === LOAD MODEL ===
print("🧠 Loading model architecture...")
model = FakeNewsShield().to(device)
print("✅ Model loaded")

# === TRAINING SETUP ===
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
criterion = nn.CrossEntropyLoss()

print("🔄 Starting training... (This will take 8-20 minutes on CPU)")

for epoch in range(4):
    model.train()
    total_loss = 0.0
    
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)
        
        outputs = model(input_ids, pixel_values)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if i % 8 == 0:
            print(f"   Epoch {epoch+1}/4 | Batch {i}/{len(dataloader)} | Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    print(f"✅ Epoch {epoch+1}/4 Completed | Average Loss: {avg_loss:.4f}")

# === SAVE MODEL ===
torch.save(model.state_dict(), "models/fakenewsshield_model.pth")
print("🎉 TRAINING COMPLETE! Model saved successfully.")
print("You can now run: streamlit run app.py")