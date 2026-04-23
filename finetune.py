from transformers import RobertaTokenizerFast
import torch

from src.v1.emocoder import EmoCoderCore
from src.v1.class_head import ClassificationHead
from config import ModelConfig, FinetuneConfig


ENCODER_PATH = "models/v1/pretrain"

train_config = FinetuneConfig(num_labels=28)


tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

with open(f"{ENCODER_PATH}/model_config.json", "r") as f:
    model_config = ModelConfig.model_validate_json(f.read())

encoder = EmoCoderCore(model_config)



if "checkpoints" in ENCODER_PATH:
    from safetensors.torch import load_file

    raw_state_dict = load_file(f"{ENCODER_PATH}/model.safetensors")
    
    encoder_state_dict = {
        k.replace("encoder.", ""): v 
        for k, v in raw_state_dict.items() 
        if k.startswith("encoder.")
    }
    
    encoder.load_state_dict(encoder_state_dict, strict=True)

else:
    encoder.load_state_dict(torch.load(f"{ENCODER_PATH}/encoder_only.pth"), strict=True)


model = ClassificationHead(encoder, model_config, train_config.num_labels)




# Rozdělení parametrů
encoder_params = model.encoder.parameters()
head_params = model.classifier.parameters()

optimizer = torch.optim.AdamW([
    {'params': encoder_params, 'lr': 2e-5}, # Jemné doladění
    {'params': head_params, 'lr': 1e-3}    # Agresivní učení hlavy
], weight_decay=0.01)