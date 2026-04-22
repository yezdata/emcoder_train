import torch.nn as nn
import torch
from emocoder import EmoCoderCore
from config import ModelConfig
from types_shapes import BatchIDs, Logits, MeanPooled, TokenEmbeds


class ClassificationHead(nn.Module):
    def __init__(self, encoder: EmoCoderCore, config: ModelConfig, n_labels: int):
        super().__init__()
        self.encoder = encoder

        self.class_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, n_labels)
        )
    

    def set_mc_dropout(self, active: bool = True):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train(active)


    @staticmethod
    def _masked_mean_pooling(features: TokenEmbeds, mask: torch.Tensor) -> MeanPooled:
        # features: (B, S, D)
        # mask: (B, S)
        mask = mask.unsqueeze(-1)  # (B, S, 1)
        masked_features = features * mask  # (B, S, D)
        sum_masked_features = masked_features.sum(dim=1)  # (B, D)
        count_tokens = torch.clamp(mask.sum(dim=1), min=1e-9)  # (B, 1)
        return sum_masked_features / count_tokens  # (B, D)


    def forward(self, x: BatchIDs, mask: torch.Tensor) -> Logits:
        features = self.encoder(x, mask)
        # mean pooling + padding mask
        pooled = self._masked_mean_pooling(features, mask) # (B, S, D) -> (B, D)
        return self.class_head(pooled)