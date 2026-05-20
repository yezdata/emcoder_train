import torch.nn as nn

from ..config_models import EmCoderEncoderConfig
from ..types_shapes import BatchIDs, BatchMask, MLMLogits
from ..utils import RMSNorm
from .backbone_encoder import EmCoderEncoder


class MLMHead(nn.Module):
    def __init__(self, config: EmCoderEncoderConfig):
        super().__init__()

        self.encoder = EmCoderEncoder(config)

        self.transform = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            RMSNorm(config.d_model),
        )

        self.decoder = nn.Linear(config.d_model, config.vocab_size)
        # Weight tieing
        self.decoder.weight = self.encoder.token_embedding.weight

    def forward(self, x: BatchIDs, mask: BatchMask) -> MLMLogits:
        features = self.encoder(x, mask)
        features = self.transform(features)
        return self.decoder(features)
