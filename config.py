from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    """
    'small_scratch': {
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'ffn_dim': 2048,
        'dropout': 0.1,
        'embed_dropout': 0.1,
        'mc_dropout_rate': 0.3,
    },
    'medium_scratch': {
        'd_model': 768,
        'n_heads': 12,
        'n_layers': 8,
        'ffn_dim': 3072,
        'dropout': 0.1,
        'embed_dropout': 0.1,
        'mc_dropout_rate': 0.3,
    },
    """
    vocab_size: int
    max_seq_len: int
    
    d_model: int
    n_head: int
    n_layers: int
    # TODO: d_ffn = d_model * 4?
    d_ffn: int
    
    dropout: float

    model_config = SettingsConfigDict(env_file=".env", env_prefix="MODEL_")