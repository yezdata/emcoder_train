import torch
import torch.nn as nn
from tqdm import tqdm
from accelerate import Accelerator
from transformers import RobertaTokenizerFast, DataCollatorForLanguageModeling, get_cosine_schedule_with_warmup
from datasets import load_from_disk

from src.config_models import ModelConfig, PretrainConfig
from src.types_shapes import MLMLabels, MLMLogits
from src.utils import save_encoder, init_weights
from src.v1.emcoder_core import EmCoderCore
from src.v1.mlm_head import MLMHead


torch.manual_seed(1337)



def compute_loss(criterion: nn.CrossEntropyLoss, logits: MLMLogits, labels: MLMLabels) -> torch.Tensor:
    B, S, V = logits.shape
    return criterion(logits.view(B * S, V), labels.view(-1))


def evaluate(criterion: nn.CrossEntropyLoss, model: MLMHead, dataloader: torch.utils.data.DataLoader) -> float:
    model.eval()
    total_loss = 0.0
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(batch["input_ids"], batch["attention_mask"])
            loss = compute_loss(criterion, outputs, batch["labels"])
        
        total_loss += loss.item()

    model.train()
    return total_loss / len(dataloader)

    



def main(save_path: str) -> None:
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")


    model_config = ModelConfig(vocab_size=tokenizer.vocab_size, max_seq_len=512, d_model=768, n_head=12, d_ffn=3072, n_layers=6, dropout=0.1)

    train_config = PretrainConfig(tokenized_ds_dir=f"data/openwebtext_v1_seq{model_config.max_seq_len}", lr=3e-4, batch_size=64, gradient_accumulation_steps=32, num_epochs=3, lr_warmup=0.05, weight_decay=0.01)

    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=train_config.gradient_accumulation_steps)




    encoder = EmCoderCore(model_config)
    model = MLMHead(encoder, model_config)

    model.apply(init_weights)



    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=0.15
    )

    tokenized_train_ds = load_from_disk(f"{train_config.tokenized_ds_dir}/train")
    tokenized_eval_ds = load_from_disk(f"{train_config.tokenized_ds_dir}/eval")

    train_dataloader = torch.utils.data.DataLoader(tokenized_train_ds, batch_size=train_config.batch_size, collate_fn=data_collator, shuffle=True, num_workers=8, pin_memory=True)
    eval_dataloader = torch.utils.data.DataLoader(tokenized_eval_ds, batch_size=train_config.batch_size, collate_fn=data_collator, shuffle=False, num_workers=8, pin_memory=True)



    total_steps = (len(train_dataloader) * train_config.num_epochs) // train_config.gradient_accumulation_steps

    optimizer = torch.optim.AdamW(model.parameters(), fused=True, lr=train_config.lr, weight_decay=train_config.weight_decay)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * train_config.lr_warmup),
        num_training_steps=total_steps
    )


    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )



    # TRAIN LOOP
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    model.train()

    for epoch in range(train_config.num_epochs):
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            with accelerator.accumulate(model):
                outputs = model(batch["input_ids"], batch["attention_mask"])
                loss = compute_loss(criterion, outputs, batch["labels"])

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if step % 1000 == 0:
                accelerator.print(f"Epoch {epoch + 1} | Step {step} | TRAIN  Loss: {loss.item():.4f}")

            if step % 10000 == 0 and step > 0:
                eval_loss = evaluate(criterion, model, eval_dataloader)
                accelerator.print(f"TRAIN Loss: {loss.item():.4f} | Eval Loss: {eval_loss:.4f}")

                step_path = f"{save_path}/checkpoints/epoch_{epoch + 1}/step_{step}"

                save_encoder(accelerator, model, step_path, model_config, train_config, loss.item(), eval_loss)

        eval_loss = evaluate(criterion, model, eval_dataloader)
        accelerator.print(f"FINISH EPOCH {epoch + 1} | TRAIN Loss: {loss.item():.4f} | Eval Loss: {eval_loss:.4f}")


    save_encoder(accelerator, model, save_path, model_config, train_config)



if __name__ == '__main__':
    SAVE_PATH = "models/v1/pretrain"
    

    main(SAVE_PATH)