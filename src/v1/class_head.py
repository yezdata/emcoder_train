import torch.nn as nn
import torch

from config import ClassifierConfig
from .emocoder import EmoCoderCore
from ..types_shapes import BatchIDs, BatchMask, ClassLogits, ClassLogitsMC, MeanPooled, TokenEmbeds


class ClassificationHead(nn.Module):
    def __init__(self, encoder: EmoCoderCore, config: ClassifierConfig, n_labels: int):
        super().__init__()

        self.encoder = encoder

        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, n_labels)
        )
    

    def _set_mc_dropout(self, active: bool = True):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train(active)


    @staticmethod
    def _masked_mean_pooling(features: TokenEmbeds, mask: BatchMask) -> MeanPooled:
        mask = mask.unsqueeze(-1)  # (B, S, 1)
        masked_features = features * mask  # (B, S, D)
        sum_masked_features = masked_features.sum(dim=1)  # (B, D)
        count_tokens = torch.clamp(mask.sum(dim=1), min=1e-9)  # (B, 1)
        return sum_masked_features / count_tokens  # (B, D)


    def mc_forward(self, x: BatchIDs, mask: BatchMask, n_samples: int) -> ClassLogitsMC:
        self._set_mc_dropout(active=True)

        B, S = x.shape
        x_stacked = x.repeat(n_samples, 1) # (n_samples * B, S)
        mask_stacked = mask.repeat(n_samples, 1)

        features = self.encoder(x_stacked, mask_stacked)
        pooled = self._masked_mean_pooling(features, mask_stacked)
        logits = self.classifier(pooled) # (n_samples * B, n_labels)

        return logits.view(n_samples, B, -1)


    def forward(self, x: BatchIDs, mask: BatchMask) -> ClassLogits:
        features = self.encoder(x, mask)

        pooled = self._masked_mean_pooling(features, mask)
        return self.classifier(pooled)