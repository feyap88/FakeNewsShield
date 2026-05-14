from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
from PIL import Image
import torch
from transformers import RobertaTokenizer, ViTImageProcessor
import requests
from io import BytesIO

class MultimodalFakeNewsDataset(Dataset):
    def __init__(self, split="train", max_samples=300):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.max_samples = max_samples
        
        datasets_list = []
        
        try:
            gossip = load_dataset("Jinyan1/GossipCop", split=split)
            politi = load_dataset("Jinyan1/PolitiFact", split=split)
            datasets_list.extend([gossip, politi])
            print("✅ Loaded GossipCop + PolitiFact")
        except Exception as e:
            print(f"⚠️ Gossip/PolitiFact failed: {e}")
        
        try:
            fakeddit = load_dataset("Shivanshu/fakeddit", "all", split=split)
            datasets_list.append(fakeddit)
            print("✅ Loaded Fakeddit")
        except Exception as e:
            print(f"⚠️ Fakeddit failed: {e}")
        
        if datasets_list:
            self.dataset = concatenate_datasets(datasets_list)
            print(f"✅ Combined real datasets: {len(self.dataset)} samples")
        else:
            print("🔄 Using enhanced synthetic data")
            self.dataset = self._create_enhanced_synthetic(max_samples)
        
        # Handle both HF Dataset and list
        if hasattr(self.dataset, "shuffle"):
            self.dataset = self.dataset.shuffle(seed=42).select(range(min(max_samples, len(self.dataset))))
        else:
            # For synthetic list
            import random
            random.seed(42)
            random.shuffle(self.dataset)
            self.dataset = self.dataset[:min(max_samples, len(self.dataset))]
        
        print(f"✅ Final dataset ready with {len(self.dataset)} samples")

    def _create_enhanced_synthetic(self, num_samples):
        data = []
        for i in range(num_samples):
            is_real = i % 3 != 0
            if is_real:
                text = "Official report: Kenya launches new digital learning program backed by government sources."
                color = 'green'
            else:
                text = "Shocking breaking news! Aliens landed in Nairobi yesterday with no evidence!"
                color = 'red'
            img = Image.new('RGB', (224, 224), color=color)
            data.append({"text": text, "image": img, "label": 0 if is_real else 1})
        return data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        text = item.get("text") or item.get("claim") or "No text"
        inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        
        image = item.get("image")
        if not isinstance(image, Image.Image):
            image = Image.new('RGB', (224, 224), color='gray')
        
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        
        label = torch.tensor(item.get("label", 0), dtype=torch.long)
        
        return {
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "pixel_values": pixel_values,
            "label": label
        }