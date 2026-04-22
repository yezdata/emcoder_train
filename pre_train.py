from transformers import RobertaTokenizerFast
from config import ModelConfig
from datasets import load_dataset


model_config = ModelConfig()

torch.manual_seed(1337)

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

model_config.vocab_size = tokenizer.vocab_size
pad_token_id




ds = load_dataset("Skylion007/openwebtext")




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

model.apply(init_weights)


scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=3000,    # hodně warmup pro scratch
    num_training_steps=total_steps
)



optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)



encoder = MyEncoder(config)
model = MyModelForMLM(encoder, config)
# Trénuj MLM...
torch.save(encoder.state_dict(), "encoder_weights.pth")


# TODO save model, modelconfig json dump
# - load modelconfig from json pak