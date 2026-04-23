import torch.nn as nn

from config import ModelConfig
from .emocoder import EmoCoderCore
from ..types_shapes import BatchIDs, BatchMask, MLMLogits


class MLMHead(nn.Module):
    def __init__(self, encoder: EmoCoderCore, config: ModelConfig):
        super().__init__()

        self.encoder = encoder

        self.transform = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model)
        )

        self.decoder = nn.Linear(config.d_model, config.vocab_size)
        # Weight tieing
        self.decoder.weight = self.encoder.token_embedding.weight
        
    def forward(self, x: BatchIDs, mask: BatchMask) -> MLMLogits:
        features = self.encoder(x, mask)
        features = self.transform(features)
        return self.decoder(features)