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


class ConcreteDropout(nn.Module):
    """
    Concrete Dropout for linear layers
    Approximate optimal dropout rate 'p' with variational inference
    """

    def __init__(
        self,
        layer: nn.Module,
        weight_regularizer: float = 1e-5,
        dropout_regularizer: float = 1e-5,
        init_min: float = 0.1,
        init_max: float = 0.1,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.layer = layer
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.temperature = temperature

        # Init trainable param p with log_alpha
        init_p = torch.empty(1).uniform_(init_min, init_max)
        self.log_alpha = nn.Parameter(torch.log(init_p / (1.0 - init_p)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute p with sigmoid func from log_alpha
        p = torch.sigmoid(self.log_alpha)

        if self.training:
            # Gumbel-Softmax relaxation for continuous sampling of the mask
            eps = 1e-7
            u = torch.empty_like(x).uniform_(0, 1)
            concrete_prob = torch.sigmoid(
                (
                    torch.log(p + eps)
                    - torch.log(1.0 - p + eps)
                    + torch.log(u + eps)
                    - torch.log(1.0 - u + eps)
                )
                / self.temperature
            )

            # Apply dropout mask to the input
            drop_mask = 1.0 - concrete_prob
            x = x * drop_mask / (1.0 - p + eps)
            return self.layer(x)
        else:
            # Standard eval mode - without dropout
            return self.layer(x)

    def mc_forward_layer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Manual Stochastic forwards pass
            - Uses Gumbel-Softmax relaxation to sample dropout masks during training
            - Uses standard Bernoulli dropout mask during evaluation for evaluation / inference
        """
        p = torch.sigmoid(self.log_alpha)
        eps = 1e-7

        if self.training:
            u = torch.empty_like(x).uniform_(0, 1)
            concrete_prob = torch.sigmoid(
                (
                    torch.log(p + eps)
                    - torch.log(1.0 - p + eps)
                    + torch.log(u + eps)
                    - torch.log(1.0 - u + eps)
                )
                / self.temperature
            )
            drop_mask = 1.0 - concrete_prob
        else:
            drop_mask = (torch.rand_like(x) > p).to(dtype=x.dtype)

        x = x * drop_mask / (1.0 - p + eps)
        return self.layer(x)

    def get_regularization_loss(self, input_dim: int) -> torch.Tensor:
        """Compute KL divergence regularization for the given layer"""
        p = torch.sigmoid(self.log_alpha)

        # L2 regularization on weights, scaled by dropout rate
        weight = self.layer.weight
        weight_reg = (
            self.weight_regularizer * torch.sum(torch.square(weight)) / (1.0 - p + 1e-7)
        )

        # Entropy penalty for dropout to prevent collapse to p=1 or p=0
        dropout_reg = p * torch.log(p + 1e-7) + (1.0 - p) * torch.log(1.0 - p + 1e-7)
        dropout_reg = self.dropout_regularizer * input_dim * dropout_reg

        return weight_reg + dropout_reg


class EmCoder(nn.Module):
    """The full EmCoder model, including the backbone encoder and the classification head."""

    def __init__(
        self,
        config: EmCoderConfig,
        num_train_examples: int | None = None,
        pos_weights: torch.Tensor | None = None,
    ):
        super().__init__()

        self.encoder = EmCoderEncoder(config)

        w_reg = 0.0
        d_reg = 0.0

        if num_train_examples is not None and pos_weights is not None:
            N = num_train_examples
            C = config.num_labels
            mean_pos_weight = pos_weights.mean().item()

            length_scale = 1e-2
            tau = mean_pos_weight / C

            w_reg = (length_scale**2) / (2 * tau * N)
            d_reg = 1 / (2 * tau * N)

        self.linear1 = nn.Linear(config.d_model, config.d_model)
        self.gelu = nn.GELU()
        self.concrete_drop = ConcreteDropout(
            nn.Linear(config.d_model, config.num_labels),
            weight_regularizer=w_reg,
            dropout_regularizer=d_reg,
            init_min=config.dropout,
            init_max=config.dropout,
        )

        # self.classifier = nn.Sequential(
        #     nn.Linear(config.d_model, config.d_model),
        #     nn.GELU(),
        #     nn.Dropout(config.dropout),
        #     nn.Linear(config.d_model, config.num_labels),
        # )

    def get_cd_loss(self, d_model: int) -> torch.Tensor:
        """Add up the regularization losses from all Concrete Dropout modules."""
        return self.concrete_drop.get_regularization_loss(input_dim=d_model)

    def _set_mc_dropout(self, active: bool = True):
        for m in self.encoder.modules():
            if isinstance(m, nn.Dropout):
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
        num_labels = self.concrete_drop.layer.out_features

        all_logits = torch.empty((n_samples, B, num_labels), device=x.device)

        is_training = self.training
        self._set_mc_dropout(active=True)

        try:
            with torch.no_grad():
                for i in range(0, n_samples, max_batch_size):
                    batch_samples = min(max_batch_size, n_samples - i)

                    x_stacked = x.repeat(batch_samples, 1)  # (batch_samples * B, S)
                    mask_stacked = mask.repeat(
                        batch_samples, 1
                    )  # (batch_samples * B, S)

                    features = self.encoder(
                        x_stacked, mask_stacked
                    )  # (batch_samples * B, S, D)
                    pooled = self._masked_mean_pooling(features, mask_stacked)

                    # manual forced stochastic forward pass for classification head
                    h1 = self.gelu(self.linear1(pooled))
                    logits = self.concrete_drop.mc_forward_layer(h1)

                    all_logits[i : i + batch_samples] = logits.view(
                        batch_samples, B, -1
                    )
        finally:
            self._set_mc_dropout(active=is_training)

        return all_logits

    def forward(self, x: BatchIDs, mask: BatchMask) -> ClassLogits:
        """Standard forward pass without MC Dropout."""
        features = self.encoder(x, mask)
        pooled = self._masked_mean_pooling(features, mask)

        # standard forward pass through classification head
        h1 = self.gelu(self.linear1(pooled))
        return self.concrete_drop(h1)
