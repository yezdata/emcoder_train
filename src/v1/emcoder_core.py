import torch
import torch.nn as nn

from ..config_models import ModelConfig
from ..types_shapes import BatchIDs, BatchMask, TokenEmbeds


class EmCoderCore(nn.Module):
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
        
        self.final_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, x: BatchIDs, mask: BatchMask) -> TokenEmbeds:
        seq_len = x.size(1)
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)

        x = self.token_embedding(x) + self.pos_embedding(pos_ids)

        x = self.embed_norm(x)
        x = self.dropout(x)

        padding_mask = (mask == 0)

        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.final_norm(encoded)