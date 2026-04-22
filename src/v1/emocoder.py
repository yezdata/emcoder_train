import torch.nn as nn
import torch
from config import ModelConfig
from types_shapes import BatchIDs, TokenEmbeds


class EmoCoderCore(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model
        )
        self.pos_embedding = nn.Embedding(
            config.max_seq_len,
            config.d_model
        )

        self.embed_norm = nn.LayerNorm(config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.d_ffn,
            dropout=config.dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.n_layers
        )
        
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, x: BatchIDs, mask: torch.Tensor) -> TokenEmbeds:
        # (B, S, D)
        # TODO
        return x