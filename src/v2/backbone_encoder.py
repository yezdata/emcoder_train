import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

from ..config_models import EmCoderEncoderConfig
from ..types_shapes import BatchIDs, BatchMask, TokenEmbeds
from ..utils import RMSNorm, SwiGLU


class EmCoderEncoderLayer(nn.Module):
    """Custom Pre-LN Transformer Encoder Layer with RoPE and FlashAttention."""

    def __init__(self, config: EmCoderEncoderConfig, rope: RotaryEmbedding):
        super().__init__()
        self.n_head = config.n_head
        self.d_head = config.d_model // config.n_head
        self.rope = rope

        # Attention projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.ln1 = RMSNorm(config.d_model)
        self.ln2 = RMSNorm(config.d_model)

        self.ffn = SwiGLU(config.d_model, config.d_ffn)

        self.dropout = nn.Dropout(config.dropout)

        # mark for initialization
        self.out_proj._is_residual = True
        self.ffn.wo._is_residual = True

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # MULTI-HEAD ATTENTION
        residual = x
        nx = self.ln1(x)
        B, S, _ = nx.shape

        # Projections -> (B, H, S, D_head)
        q = self.q_proj(nx).view(B, S, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(nx).view(B, S, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(nx).view(B, S, self.n_head, self.d_head).transpose(1, 2)

        q = self.rope.rotate_queries_or_keys(q)
        k = self.rope.rotate_queries_or_keys(k)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.dropout.training else 0.0,
        )

        # Join heads -> (B, S, D_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        x = residual + self.dropout(self.out_proj(attn_out))

        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class EmCoderEncoder(nn.Module):
    """The core encoder architecture of EmCoder Transformer."""

    def __init__(self, config: EmCoderEncoderConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_norm = RMSNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.rope = RotaryEmbedding(dim=config.d_model // config.n_head)

        self.layers = nn.ModuleList(
            [EmCoderEncoderLayer(config, self.rope) for _ in range(config.n_layers)]
        )

        self.final_norm = RMSNorm(config.d_model)

    def forward(self, x: BatchIDs, mask: BatchMask) -> TokenEmbeds:
        """Standard forward pass through the encoder."""
        x = self.token_embedding(x)
        x = self.embed_norm(x)
        x = self.dropout(x)

        # attn_mask = mask.view(x.size(0), 1, 1, x.size(1)).to(dtype=torch.bool)
        B, S = mask.shape
        attn_mask = mask.view(B, 1, 1, S).expand(B, 1, S, S).to(dtype=torch.bool)

        for layer in self.layers:
            x = layer(x, attn_mask)

        return self.final_norm(x)


# NOTE WITHOUT SwiGLU / GeGLU AND RMSNorm

# class EmCoderEncoderLayer(nn.Module):
#     """Custom Pre-LN Transformer Encoder Layer with RoPE and FlashAttention."""

#     def __init__(self, config: EmCoderEncoderConfig, rope: RotaryEmbedding):
#         super().__init__()
#         self.n_head = config.n_head
#         self.d_head = config.d_model // config.n_head
#         self.rope = rope

#         # Attention projections
#         self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
#         self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
#         self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
#         self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

#         self.ln1 = nn.LayerNorm(config.d_model)
#         self.ln2 = nn.LayerNorm(config.d_model)

#         self.ffn = nn.Sequential(
#             nn.Linear(config.d_model, config.d_ffn),
#             nn.GELU(),
#             nn.Dropout(config.dropout),
#             nn.Linear(config.d_ffn, config.d_model),
#             nn.Dropout(config.dropout),
#         )
#         self.dropout = nn.Dropout(config.dropout)

#     def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
#         # MULTI-HEAD ATTENTION
#         residual = x
#         nx = self.ln1(x)
#         B, S, _ = nx.shape

#         # Projection + permutation -> (B, H, S, D)
#         q = self.q_proj(nx).view(B, S, self.n_head, self.d_head).transpose(1, 2)
#         k = self.k_proj(nx).view(B, S, self.n_head, self.d_head).transpose(1, 2)
#         v = self.v_proj(nx).view(B, S, self.n_head, self.d_head).transpose(1, 2)

#         q = self.rope.rotate_queries_or_keys(q)
#         k = self.rope.rotate_queries_or_keys(k)

#         attn_out = F.scaled_dot_product_attention(
#             q,
#             k,
#             v,
#             attn_mask=attn_mask,
#             dropout_p=self.dropout.p if self.dropout.training else 0.0,
#         )

#         # Join heads -> (B, S, D_model)
#         attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
#         x = residual + self.dropout(self.out_proj(attn_out))

#         x = x + self.ffn(self.ln2(x))
#         return x


# class EmCoderEncoder(nn.Module):
#     """The core encoder architecture of EmCoder Transformer."""

#     def __init__(self, config: EmCoderEncoderConfig):
#         super().__init__()
#         self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
#         self.embed_norm = nn.LayerNorm(config.d_model)
#         self.dropout = nn.Dropout(config.dropout)

#         self.rope = RotaryEmbedding(dim=config.d_model // config.n_head)

#         self.layers = nn.ModuleList(
#             [EmCoderEncoderLayer(config, self.rope) for _ in range(config.n_layers)]
#         )

#         self.final_norm = nn.LayerNorm(config.d_model)

#     def forward(self, x: BatchIDs, mask: BatchMask) -> TokenEmbeds:
#         """Standard forward pass through the encoder."""
#         x = self.token_embedding(x)
#         x = self.embed_norm(x)
#         x = self.dropout(x)

#         attn_mask = mask.view(x.size(0), 1, 1, x.size(1)).bool()

#         for layer in self.layers:
#             x = layer(x, attn_mask)

#         return self.final_norm(x)
