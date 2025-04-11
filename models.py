# Transformer Block
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms import InterpolationMode
DROPOUT = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=True)[0]
        x = x + self.mlp(self.norm2(x))
        return x

# Masked Autoencoder (MAE)
class MaskedAutoencoder(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, num_layers, mask_ratio=0.75):
        super().__init__()
        self.feature_extractor = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).features
        self.fc = nn.Linear(1280, embed_dim)

        self.mask_ratio = mask_ratio
        self.transformer = nn.Sequential(*[TransformerBlock(embed_dim, num_heads, hidden_dim, DROPOUT) for _ in range(num_layers)])
        self.decoder = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        features = self.fc(features)

        mask = torch.rand(features.shape, device=device) > self.mask_ratio
        masked_features = features * mask

        encoded = self.transformer(masked_features.unsqueeze(1))
        reconstructed = self.decoder(encoded.squeeze(1))
        return reconstructed, features
# Pretraining MAE
def pretrain_mae(mae, train_loader, epochs=10):
    optimizer = torch.optim.AdamW(mae.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()
    mae.train()

    for epoch in range(epochs):
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{epochs}")
        for images, _ in loop:
            images = images.to(device)
            optimizer.zero_grad()
            reconstructed, original = mae(images)
            loss = criterion(reconstructed, original)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# Classification Model
class EnhancedViT(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, hidden_dim, num_classes):
        super().__init__()
        self.mae = MaskedAutoencoder(embed_dim, num_heads, hidden_dim, num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        _, features = self.mae(x)
        return self.fc(features)
