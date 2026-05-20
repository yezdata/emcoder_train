import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from pydantic_settings import BaseSettings
from safetensors.torch import save_file


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.wi = nn.Linear(d_model, 2 * d_ffn, bias=False)
        self.wo = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.wi(x).chunk(2, dim=-1)
        return self.wo(x1 * F.silu(x2))


class GeGLU(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.wi = nn.Linear(d_model, 2 * d_ffn, bias=False)
        self.wo = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.wi(x).chunk(2, dim=-1)
        return self.wo(x1 * F.gelu(x2))


def init_weights_modern(module: nn.Module, n_layers: int):
    """Modern initialization for Transformers (RMSNorm, GeGLU / SwiGLU)."""

    if isinstance(module, nn.Linear):
        # scale down the init for residual connections
        if getattr(module, "_is_residual", False):
            std = 0.02 / ((2 * n_layers) ** 0.5)
        else:
            std = 0.02

        nn.init.trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02)

    elif isinstance(module, RMSNorm):
        nn.init.ones_(module.weight)


def init_weights(module: nn.Module):
    """Standard initialization for Transformers (LayerNorm, Linear)."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def save_emcoder_encoder(
    accelerator: Accelerator,
    model: nn.Module,
    save_path: str,
    model_config: BaseSettings,
    train_config: BaseSettings,
    loss: float | None = None,
    eval_loss: float | None = None,
) -> None:
    os.makedirs(save_path, exist_ok=True)

    resume_dir = os.path.join(save_path, "accelerator_state")
    accelerator.save_state(resume_dir, safe_serialization=True)

    # SAVE ONLY ENCODER
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save(unwrapped_model.encoder.state_dict(), f"{save_path}/encoder_only.pth")

    with open(f"{save_path}/config.json", "w") as f:
        f.write(model_config.model_dump_json(indent=4))
    with open(f"{save_path}/train_config.json", "w") as f:
        f.write(train_config.model_dump_json(indent=4))

    if loss is not None and eval_loss is not None:
        with open(f"{save_path}/train_state.json", "w") as f:
            json.dump({"train_loss": loss, "eval_loss": eval_loss}, f, indent=4)


def save_emcoder(
    accelerator: Accelerator,
    model: nn.Module,
    save_path: str,
    model_config: BaseSettings,
    train_config: BaseSettings,
    loss: float | None = None,
    eval_loss: float | None = None,
) -> None:
    os.makedirs(save_path, exist_ok=True)

    # SAVE ONLY WHOLE MODEL
    unwrapped_model = accelerator.unwrap_model(model)
    save_file(unwrapped_model.state_dict(), f"{save_path}/model.safetensors")

    with open(f"{save_path}/config.json", "w") as f:
        f.write(model_config.model_dump_json(indent=4))
    with open(f"{save_path}/train_config.json", "w") as f:
        f.write(train_config.model_dump_json(indent=4))

    if loss is not None and eval_loss is not None:
        with open(f"{save_path}/train_state.json", "w") as f:
            json.dump({"train_loss": loss, "eval_loss": eval_loss}, f, indent=4)
