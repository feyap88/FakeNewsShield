import torch
import torch.nn as nn
from transformers import RobertaModel, ViTModel

class FakeNewsShield(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        self.image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        self.text_proj = nn.Linear(768, 512)
        self.image_proj = nn.Linear(768, 512)
        
        # Cross Attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(256, 2)  # 0 = REAL, 1 = FAKE
        
    def forward(self, input_ids, pixel_values):
        # Text features
        text_out = self.text_encoder(input_ids=input_ids).pooler_output
        # Image features
        image_out = self.image_encoder(pixel_values=pixel_values).pooler_output
        
        text_emb = self.text_proj(text_out)
        image_emb = self.image_proj(image_out)
        
        # Cross-modal attention
        attn_output, _ = self.cross_attn(text_emb.unsqueeze(1), image_emb.unsqueeze(1), image_emb.unsqueeze(1))
        attn_output = attn_output.squeeze(1)
        
        # Fusion
        fused = torch.cat([text_emb, attn_output], dim=1)
        fused = self.fusion(fused)
        
        logits = self.classifier(fused)
        return logits