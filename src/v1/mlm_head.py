import torch.nn as nn

from config import ModelConfig
from .emocoder import EmoCoderCore
from ..types_shapes import BatchIDs, BatchMask, MLMLogits


class MLMHead(nn.Module):
    def __init__(self, encoder: EmoCoderCore, config: ModelConfig):
        super().__init__()

        self.encoder = encoder
        self.mlm_head = nn.Linear(config.d_model, config.vocab_size)
        # Weight tieing
        self.mlm_head.weight = self.encoder.token_embedding.weight
        
    def forward(self, x: BatchIDs, mask: BatchMask) -> MLMLogits:
        features = self.encoder(x, mask)
        return self.mlm_head(features)