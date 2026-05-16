import torch
import torch.nn as nn

from ..config_models import ClassifierConfig
from ..types_shapes import (
    BatchIDs,
    BatchMask,
    ClassLogits,
    ClassLogitsMC,
    MeanPooled,
    TokenEmbeds,
)
from .emcoder_core import EmCoderCore


class ClassificationHead(nn.Module):
    """The full EmCoder model, including the classification head."""

    def __init__(self, encoder: EmCoderCore, config: ClassifierConfig):
        super().__init__()

        self.encoder = encoder

        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_labels),
        )

    def _set_mc_dropout(self, active: bool = True):
        # for m in self.modules():
        for m in self.classifier.modules():
            if isinstance(m, nn.Dropout):
                m.train(active)

    @staticmethod
    def _masked_mean_pooling(features: TokenEmbeds, mask: BatchMask) -> MeanPooled:
        mask = mask.unsqueeze(-1)  # (B, S, 1)
        masked_features = features * mask  # (B, S, D)
        sum_masked_features = masked_features.sum(dim=1)  # (B, D)
        count_tokens = torch.clamp(mask.sum(dim=1), min=1e-9)  # (B, 1)
        return sum_masked_features / count_tokens  # (B, D)

    # TODO argument: neco jako grad accumulation -> aby se nemusela do VRAM cpat cela N_samples ale slo to napr po batch sekvencne
    # TODO: pridat novou metodu do HF + pridat ten docstring do toho .py
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
        self._set_mc_dropout(active=True)

        if max_batch_size is None:
            max_batch_size = n_samples

        B, S = x.shape
        num_labels = self.classifier[-1].out_features

        all_logits = torch.empty((n_samples, B, num_labels), device=x.device)

        features = self.encoder(x, mask)  # (B, S, D)

        for i in range(0, n_samples, max_batch_size):
            batch_samples = min(max_batch_size, n_samples - i)

            features_stacked = features.repeat(
                batch_samples, 1, 1
            )  # (n_samples * B, S, D)
            mask_stacked = mask.repeat(batch_samples, 1)

            pooled = self._masked_mean_pooling(features_stacked, mask_stacked)
            logits = self.classifier(pooled)  # (n_samples * B, num_labels)

            all_logits[i : i + batch_samples] = logits.view(batch_samples, B, -1)

        return all_logits

    def forward(self, x: BatchIDs, mask: BatchMask) -> ClassLogits:
        """Standard forward pass without MC Dropout."""
        features = self.encoder(x, mask)

        pooled = self._masked_mean_pooling(features, mask)
        return self.classifier(pooled)
