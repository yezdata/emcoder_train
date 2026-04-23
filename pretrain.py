import torch
import torch.nn as nn
from accelerate import Accelerator
from transformers import RobertaTokenizerFast, DataCollatorForLanguageModeling, get_cosine_schedule_with_warmup
from config import ModelConfig, TrainConfig
from datasets import load_dataset

from types_shapes import MLMLabels, MLMLogits
from v1.emocoder import EmoCoderCore
from v1.mlm_head import MLMHead


torch.manual_seed(1337)


SAVE_PATH = "models/v1/pretrain"



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



def compute_loss(logits: MLMLogits, labels: MLMLabels) -> torch.Tensor:
    B, S, V = logits.shape

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    return criterion(logits.view(B * S, V), labels.view(-1))




tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

model_config = ModelConfig(vocab_size=tokenizer.vocab_size, max_seq_len=514, d_model=768, n_head=12, d_ffn=3072, n_layers=6, dropout=0.1)

train_config = TrainConfig(lr=2e-4, batch_size=128, gradient_accumulation_steps=32, num_epochs=40, lr_warmup=0.1)


ds = load_dataset("Skylion007/openwebtext")


accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=train_config.gradient_accumulation_steps)


encoder = EmoCoderCore(model_config)
model = MLMHead(encoder, model_config)

model.apply(init_weights)



data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True, 
    mlm_probability=0.15
)
tokenized_ds = ds["train"].map(
    lambda examples: tokenizer(examples["text"], truncation=True, max_length=model_config.max_seq_len),
    batched=True,
    num_proc=16,
    remove_columns=["text"],
    keep_in_memory=True
)
train_dataloader = torch.utils.data.DataLoader(tokenized_ds, batch_size=train_config.batch_size, collate_fn=data_collator, shuffle=True, num_workers=16, pin_memory=True)



effective_batch_steps = len(train_dataloader) // train_config.gradient_accumulation_steps
total_steps = effective_batch_steps * train_config.num_epochs

optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=0.01)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * train_config.lr_warmup),
    num_training_steps=total_steps
)

model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)





# TRAIN LOOP
model.train()
for epoch in range(train_config.num_epochs):
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(model):
            optimizer.zero_grad()

            outputs = model(batch["input_ids"], batch["attention_mask"])
            loss = compute_loss(outputs, batch["labels"])

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            if accelerator.is_main_process and step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

    accelerator.wait_for_everyone()
    accelerator.save_model(model, f"{SAVE_PATH}/checkpoints/model_epoch_{epoch}.pth")







accelerator.wait_for_everyone()
accelerator.save_model(model, SAVE_PATH)

# SAVE ONLY ENCODER
unwrapped_model = accelerator.unwrap_model(model)
torch.save(unwrapped_model.encoder.state_dict(), f"{SAVE_PATH}/encoder_only.pth")

with open(f"{SAVE_PATH}/model_config.json", "w") as f:
    f.write(model_config.model_dump_json())
with open(f"{SAVE_PATH}/train_config.json", "w") as f:
    f.write(train_config.model_dump_json())