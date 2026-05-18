import torch
import torch.nn as nn

from ..config_models import EmCoderConfig
from ..types_shapes import (
    BatchIDs,
    BatchMask,
    ClassLogits,
    ClassLogitsMC,
    MeanPooled,
    TokenEmbeds,
)
from .backbone_encoder import EmCoderEncoder


class EmCoder(nn.Module):
    """The full EmCoder model, including the backbone encoder and the classification head."""

    def __init__(self, config: EmCoderConfig):
        super().__init__()

        self.encoder = EmCoderEncoder(config)

        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_labels),
        )

    def _set_mc_dropout(self, active: bool = True):
        for m in self.modules():
            if isinstance(m, nn.Dropout) or isinstance(m, nn.MultiheadAttention):
                m.train(active)

    @staticmethod
    def _masked_mean_pooling(features: TokenEmbeds, mask: BatchMask) -> MeanPooled:
        mask = mask.unsqueeze(-1)  # (B, S, 1)
        masked_features = features * mask  # (B, S, D)
        sum_masked_features = masked_features.sum(dim=1)  # (B, D)
        count_tokens = torch.clamp(mask.sum(dim=1), min=1e-9)  # (B, 1)
        return sum_masked_features / count_tokens  # (B, D)

    def mc_forward(
        self,
        x: BatchIDs,
        mask: BatchMask,
        n_samples: int,
        max_batch_size: int | None = None,
    ) -> ClassLogitsMC:
        """
        Performs Monte Carlo Dropout inference to quantify epistemic uncertainty.

        Args:
            x: Input token IDs of shape (B, S).
            mask: Attention mask of shape (B, S).
            n_samples: Total number of Monte Carlo samples.
            max_batch_size: Maximum number of samples in one forward pass.

        Returns:
            Logits of shape (n_samples, B, num_labels).
        """
        if max_batch_size is None:
            max_batch_size = n_samples

        B, S = x.shape
        num_labels = self.classifier[-1].out_features

        all_logits = torch.empty((n_samples, B, num_labels), device=x.device)

        is_training = self.training
        self._set_mc_dropout(active=True)
        try:
            for i in range(0, n_samples, max_batch_size):
                batch_samples = min(max_batch_size, n_samples - i)

                x_stacked = x.repeat(batch_samples, 1)  # (batch_samples * B, S)
                mask_stacked = mask.repeat(batch_samples, 1)  # (batch_samples * B, S)

                features = self.encoder(
                    x_stacked, mask_stacked
                )  # (batch_samples * B, S, D)

                pooled = self._masked_mean_pooling(features, mask_stacked)
                logits = self.classifier(pooled)  # (n_samples * B, num_labels)

                all_logits[i : i + batch_samples] = logits.view(batch_samples, B, -1)
        finally:
            self._set_mc_dropout(active=is_training)

        return all_logits

    def forward(self, x: BatchIDs, mask: BatchMask) -> ClassLogits:
        """Standard forward pass without MC Dropout."""
        features = self.encoder(x, mask)
        pooled = self._masked_mean_pooling(features, mask)
        return self.classifier(pooled)
