import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.types_shapes import ClassLabels, ClassLogits


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(
        self, logits: ClassLogits, labels: ClassLabels, reduction: str | None = "mean"
    ) -> torch.Tensor:
        labels = labels.to(logits.dtype)

        probs = torch.sigmoid(logits)

        log_pos = F.logsigmoid(logits)
        pos_weight = torch.exp(F.logsigmoid(-logits) * self.gamma_pos)
        loss_pos = -labels * pos_weight * log_pos

        pm = torch.clamp(probs - self.clip, min=0.0)

        neg_weight = (pm + self.eps).pow(self.gamma_neg)

        log_neg_shifted = torch.log(1.0 - pm + self.eps)

        loss_neg = -(1.0 - labels) * neg_weight * log_neg_shifted

        loss = loss_pos + loss_neg

        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        return loss

    def compute_bayesian_loss(
        self, logits_samples: torch.Tensor, labels: ClassLabels
    ) -> torch.Tensor:
        n_samples, batch_size, num_classes = logits_samples.shape
        labels_expanded = labels.unsqueeze(0).expand(n_samples, batch_size, num_classes)

        loss_samples = self.forward(logits_samples, labels_expanded, reduction=None)

        return loss_samples.mean(dim=0).mean()


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(
        self,
        pos_counts: np.ndarray | None = None,
        neg_counts: np.ndarray | None = None,
    ):
        super().__init__()

        weights = self._compute_pos_weights(pos_counts, neg_counts)

        self.register_buffer("pos_weights", weights)

    def _compute_pos_weights(
        self, pos_counts: np.ndarray | None, neg_counts: np.ndarray | None
    ) -> torch.Tensor | None:
        if pos_counts is None or neg_counts is None:
            return None

        # USE LOG FOR NUM STABILITY
        # - 1e-5 avoids division by zero
        raw_weights = 1 + np.log((neg_counts + 1e-5) / (pos_counts + 1e-5))

        # PREVENT EXTREME WEIGTHS
        raw_weights = np.clip(raw_weights, a_min=0.1, a_max=20.0)
        return torch.tensor(raw_weights, dtype=torch.float32)

    def forward(self, logits: ClassLogits, labels: ClassLabels) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits,
            labels.to(logits.dtype),
            pos_weight=self.pos_weights,
            reduction="mean",
        )

    def compute_bayesian_loss(
        self, logits_samples: torch.Tensor, labels: ClassLabels
    ) -> torch.Tensor:
        labels = labels.to(logits_samples.dtype)

        n_samples = logits_samples.size(0)

        labels_expanded = labels.unsqueeze(0).expand(n_samples, -1, -1)

        loss_samples = F.binary_cross_entropy_with_logits(
            logits_samples,
            labels_expanded.to(logits_samples.dtype),
            pos_weight=self.pos_weights,
            reduction="none",
        )  # (n_samples, B, num_labels)
        return loss_samples.mean(dim=0).mean()
