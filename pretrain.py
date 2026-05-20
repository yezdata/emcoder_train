import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_from_disk
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)

from src.config_models import EmCoderEncoderConfig, PretrainConfig
from src.types_shapes import MLMLabels, MLMLogits
from src.utils import init_weights_modern, save_emcoder_encoder
from src.v2.mlm_head import MLMHead


def compute_loss(
    criterion: nn.CrossEntropyLoss, logits: MLMLogits, labels: MLMLabels
) -> torch.Tensor:
    B, S, V = logits.shape
    return criterion(logits.view(B * S, V), labels.view(-1))


def evaluate(
    criterion: nn.CrossEntropyLoss,
    model: MLMHead,
    dataloader: torch.utils.data.DataLoader,
    max_eval_steps: int = 100,
) -> float:
    model.eval()
    total_loss = 0.0
    steps_run = 0

    with torch.no_grad():
        for batch in dataloader:
            if steps_run >= max_eval_steps:
                break
            outputs = model(batch["input_ids"], batch["attention_mask"])
            loss = compute_loss(criterion, outputs, batch["labels"])

            total_loss += loss.detach().item()
            steps_run += 1

    return total_loss / max(steps_run, 1)


def main(save_path: str) -> None:
    set_seed(1337)

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    tokenizer.save_pretrained(save_path)

    model_config = EmCoderEncoderConfig(
        vocab_size=tokenizer.vocab_size,
    )  # type: ignore

    train_config = PretrainConfig()  # type: ignore

    accelerator = Accelerator(
        gradient_accumulation_steps=train_config.gradient_accumulation_steps
    )

    model = MLMHead(model_config)

    model.apply(lambda m: init_weights_modern(m, model_config.n_layers))

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    tokenized_train_ds = load_from_disk(f"{train_config.tokenized_ds_dir}/train")
    tokenized_eval_ds = load_from_disk(f"{train_config.tokenized_ds_dir}/eval")

    train_dataloader = torch.utils.data.DataLoader(
        tokenized_train_ds,  # type: ignore
        batch_size=train_config.batch_size,
        collate_fn=data_collator,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        tokenized_eval_ds,  # type: ignore
        batch_size=train_config.batch_size,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    total_steps = (
        len(train_dataloader) * train_config.num_epochs
    ) // train_config.gradient_accumulation_steps

    optimizer = torch.optim.AdamW(
        model.parameters(),
        fused=True,
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * train_config.lr_warmup),
        num_training_steps=total_steps,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = (
        accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    )

    # TRAIN LOOP
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    running_loss = 0.0
    running_steps = 0

    for epoch in range(train_config.num_epochs):
        model.train()

        for step, batch in tqdm(
            enumerate(train_dataloader), total=len(train_dataloader)
        ):
            with accelerator.accumulate(model):
                outputs = model(batch["input_ids"], batch["attention_mask"])
                loss = compute_loss(criterion, outputs, batch["labels"])

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                running_loss += loss.detach().item()
                running_steps += 1

            if step % 1000 == 0 and step > 0:
                avg_train_loss = running_loss / running_steps
                accelerator.print(
                    f"Epoch {epoch + 1} | Step {step} | TRAIN  Loss: {avg_train_loss:.4f}"
                )
                running_loss = 0.0
                running_steps = 0

            if step % 10000 == 0 and step > 0:
                eval_loss = evaluate(
                    criterion, model, eval_dataloader, max_eval_steps=100
                )
                accelerator.print(
                    f"CHECKPOINT {step} | TRAIN Loss: {loss.item():.4f} | Eval Loss: {eval_loss:.4f}"
                )

                step_path = f"{save_path}/epoch_{epoch + 1}/step_{step}"

                save_emcoder_encoder(
                    accelerator,
                    model,
                    step_path,
                    model_config,
                    train_config,
                    loss.item(),
                    eval_loss,
                )

                model.train()

        epoch_path = f"{save_path}/epoch_{epoch + 1}"
        save_emcoder_encoder(accelerator, model, epoch_path, model_config, train_config)

        eval_loss = evaluate(
            criterion, model, eval_dataloader, max_eval_steps=len(eval_dataloader)
        )
        accelerator.print(f"FINISH EPOCH {epoch + 1} | Eval Loss: {eval_loss:.4f}")


if __name__ == "__main__":
    SAVE_PATH = "models/v2/pretrain"

    main(SAVE_PATH)
