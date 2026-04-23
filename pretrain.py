import torch
import torch.nn as nn
from tqdm import tqdm
import json
from accelerate import Accelerator
from transformers import RobertaTokenizerFast, DataCollatorForLanguageModeling, get_cosine_schedule_with_warmup
from config import ModelConfig, TrainConfig
from datasets import load_dataset, load_from_disk

from src.types_shapes import MLMLabels, MLMLogits
from src.v1.emocoder import EmoCoderCore
from src.v1.mlm_head import MLMHead


torch.manual_seed(1337)


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


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            loss = compute_loss(outputs, labels)
        
        total_loss += loss.item()

    model.train()
    return total_loss / len(dataloader)

    



def main():
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")


    model_config = ModelConfig(vocab_size=tokenizer.vocab_size, max_seq_len=514, d_model=768, n_head=12, d_ffn=3072, n_layers=6, dropout=0.1)

    train_config = TrainConfig(save_path="models/v1/pretrain", tokenized_ds_dir=f"data/openwebtext_v1_seq{model_config.max_seq_len}", lr=1e-4, batch_size=256, gradient_accumulation_steps=16, num_epochs=40, lr_warmup=0.1)

    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=train_config.gradient_accumulation_steps)




    encoder = EmoCoderCore(model_config)
    model = MLMHead(encoder, model_config)

    model.apply(init_weights)



    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=0.15
    )

    tokenized_train_ds = load_from_disk(f"{train_config.tokenized_ds_dir}/train")
    tokenized_eval_ds = load_from_disk(f"{train_config.tokenized_ds_dir}/eval")

    train_dataloader = torch.utils.data.DataLoader(tokenized_train_ds, batch_size=train_config.batch_size, collate_fn=data_collator, shuffle=True, num_workers=16, pin_memory=True)
    eval_dataloader = torch.utils.data.DataLoader(tokenized_eval_ds, batch_size=train_config.batch_size, collate_fn=data_collator, shuffle=False, num_workers=16, pin_memory=True)



    effective_batch_steps = len(train_dataloader) // train_config.gradient_accumulation_steps
    total_steps = effective_batch_steps * train_config.num_epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=0.01)
    lr_scheduler = get_cosine_schedule_with_warmup(
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
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            with accelerator.accumulate(model):
                optimizer.zero_grad()

                outputs = model(batch["input_ids"], batch["attention_mask"])
                loss = compute_loss(outputs, batch["labels"])

                accelerator.backward(loss)
                optimizer.step()

            if not accelerator.optimizer_step_was_skipped:
                lr_scheduler.step()

            if accelerator.is_main_process and step % 1000 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

            if step % 5000 == 0 and step > 0:
                accelerator.wait_for_everyone()
                eval_loss = evaluate(model, eval_dataloader, accelerator.device)
                step_path = f"{train_config.save_path}/checkpoints/step_{step}"

                accelerator.save_model(model, step_path)
                with open(f"{step_path}/model_state.json", "w") as f:
                    json.dump({"train_loss": loss.item(), "eval_loss": eval_loss}, f)

        eval_loss = evaluate(model, eval_dataloader, accelerator.device)
        accelerator.print(f"FINISH EPOCH {epoch} | TRAIN Loss: {loss.item():.4f} | Eval Loss: {eval_loss:.4f}")









    accelerator.wait_for_everyone()
    accelerator.save_model(model, train_config.save_path)

    # SAVE ONLY ENCODER
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save(unwrapped_model.encoder.state_dict(), f"{train_config.save_path}/encoder_only.pth")

    with open(f"{train_config.save_path}/model_config.json", "w") as f:
        f.write(model_config.model_dump_json())
    with open(f"{train_config.save_path}/train_config.json", "w") as f:
        f.write(train_config.model_dump_json())



if __name__ == '__main__':
    main()