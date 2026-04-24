import torch
from torch import nn
from transformers import RobertaTokenizerFast
from safetensors.torch import save_file
from accelerate import Accelerator
import json
import os
from pydantic_settings import BaseSettings



def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)



def save_encoder(accelerator: Accelerator, model: nn.Module, save_path: str, model_config: BaseSettings, train_config: BaseSettings, loss: float | None = None, eval_loss: float | None = None) -> None:
    os.makedirs(save_path, exist_ok=True)

    accelerator.wait_for_everyone()
    accelerator.save_state(save_path, safe_serialization=True)

    # SAVE ONLY ENCODER
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save(unwrapped_model.encoder.state_dict(), f"{save_path}/encoder_only.pth")

    with open(f"{save_path}/model_config.json", "w") as f:
        f.write(model_config.model_dump_json())
    with open(f"{save_path}/train_config.json", "w") as f:
        f.write(train_config.model_dump_json())
    

    if loss and eval_loss:
        with open(f"{save_path}/model_state.json", "w") as f:
            json.dump({"train_loss": loss, "eval_loss": eval_loss}, f)


def save_emcoder(accelerator: Accelerator, model: nn.Module, save_path: str, model_config: BaseSettings, train_config: BaseSettings, tokenizer: RobertaTokenizerFast, loss: float | None = None, eval_loss: float | None = None) -> None:
    os.makedirs(save_path, exist_ok=True)

    # SAVE ONLY WHOLE MODEL
    unwrapped_model = accelerator.unwrap_model(model)
    save_file(unwrapped_model.state_dict(), f"{save_path}/model.safetensors")
    tokenizer.save_pretrained(save_path)

    with open(f"{save_path}/model_config.json", "w") as f:
        f.write(model_config.model_dump_json())
    with open(f"{save_path}/train_config.json", "w") as f:
        f.write(train_config.model_dump_json()) 
    
    if loss and eval_loss:
        with open(f"{save_path}/model_state.json", "w") as f:
            json.dump({"train_loss": loss, "eval_loss": eval_loss}, f)