from typing import Literal

from pydantic import model_validator, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    vocab_size: int
    max_seq_len: int
    
    d_model: int
    n_head: int
    n_layers: int
    d_ffn: int
    
    dropout: float

    model_config = SettingsConfigDict(env_file=".env", env_prefix="MODEL_")



class ClassifierConfig(ModelConfig):
    bayesian_train: bool
    loss_weights: Literal["sqrt", "log"] | None

    num_labels: int
    id2label: dict[int, str]
    label2id: dict[str, int]

    base_encoder_path: str


    @field_validator("id2label", mode="before")
    @classmethod
    def coerce_keys_to_int(cls, v):
        return {int(k): val for k, val in v.items()}

    @model_validator(mode='after')
    def check_consistency(self):
        if len(self.id2label) != self.num_labels:
            raise ValueError("num_labels does not match id2label dictionary len")
        return self




class PretrainConfig(BaseSettings):
    tokenized_ds_dir: str

    lr: float
    lr_warmup: float 

    weight_decay: float

    batch_size: int
    gradient_accumulation_steps: int
    num_epochs: int


    model_config = SettingsConfigDict(env_file=".env", env_prefix="PRETRAIN_")



class FinetuneConfig(BaseSettings):
    tokenized_ds_dir: str

    encoder_lr: float
    head_lr: float
    lr_warmup: float 

    weight_decay: float

    batch_size: int
    gradient_accumulation_steps: int
    num_epochs: int

    model_config = SettingsConfigDict(env_file=".env", env_prefix="FINETUNE_")