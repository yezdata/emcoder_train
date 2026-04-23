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


class PretrainConfig(BaseSettings):
    tokenized_ds_dir: str
    save_path: str 

    lr: float
    lr_warmup: float 

    batch_size: int
    gradient_accumulation_steps: int
    num_epochs: int


    model_config = SettingsConfigDict(env_file=".env", env_prefix="TRAIN_")



class FinetuneConfig(BaseSettings):
    num_labels: int