import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from typing import Literal
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import load_from_disk
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
from transformers import (
    DataCollatorWithPadding,
    RobertaTokenizerFast,  # type:ignore
    get_cosine_schedule_with_warmup,
)

from src.config_models import ClassifierConfig, FinetuneConfig, ModelConfig
from src.losses import WeightedBCEWithLogitsLoss, AsymmetricLoss
from src.utils import init_weights, save_emcoder
from src.v1.class_head import ClassificationHead
from src.v1.emcoder_core import EmCoderCore



def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    preds = (probs > 0.5).astype(int)

    return {
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "roc_auc": roc_auc_score(labels, probs, average="macro"),
    }


def evaluate_bayesian(
    criterion: AsymmetricLoss | WeightedBCEWithLogitsLoss,
    model: ClassificationHead,
    dataloader: torch.utils.data.DataLoader,
    accelerator: Accelerator,
    num_samples: int,
    max_batch_size: int
) -> float:
    model.eval()
    total_loss = 0.0
    all_pred_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc="Evaluating"):
            logits_samples = model.mc_forward(
                batch["input_ids"], batch["attention_mask"], n_samples=num_samples, max_batch_size=max_batch_size
            )

            loss = criterion.compute_bayesian_loss(logits_samples, batch["labels"])
            total_loss += loss.detach().item()

            probs = torch.sigmoid(logits_samples)
            mean_probs = probs.mean(dim=0)  # (B, num_labels)

            all_pred_probs.append(mean_probs.detach().cpu().numpy())
            all_labels.append(batch["labels"].detach().cpu().numpy())

    y_probs = np.concatenate(all_pred_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    print(compute_metrics(y_probs, y_true))

    del all_pred_probs, all_labels, y_probs, y_true
    if accelerator.device.type == "cuda":
        torch.cuda.empty_cache()

    model.train()
    return total_loss / len(dataloader)


def main(
    save_path: str,
    encoder_path: str,
    num_samples: int,
    max_batch_size: int,
) -> None:
    os.makedirs(save_path, exist_ok=True)

    encoder_lr = 2e-5
    head_lr = 1e-3
    dropout_rate = 0.15

    with open(f"{encoder_path}/config.json", "r") as f:
        encoder_config = ModelConfig.model_validate_json(f.read())

    with open("data/goemotions/emotions.txt", "r", encoding="utf-8") as f:
        label_names = [line.strip() for line in f.readlines()]

    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in enumerate(label_names)}

    model_config = ClassifierConfig(
        **encoder_config.model_dump(),
        base_encoder_path=encoder_path,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
    )
    model_config.dropout = dropout_rate

    train_config = FinetuneConfig(
        num_samples=num_samples,
        tokenized_ds_dir=f"data/goemotions_v1_seq{model_config.max_seq_len}",
        encoder_lr=encoder_lr,
        head_lr=head_lr,
        batch_size=8,
        gradient_accumulation_steps=16,
        num_epochs=5,
        lr_warmup=0.05,
        weight_decay=0.01,
    )

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
    )

    encoder = EmCoderCore(encoder_config)
    encoder.load_state_dict(
        torch.load(f"{encoder_path}/encoder_only.pth", map_location=accelerator.device),
        strict=True,
    )

    model = ClassificationHead(encoder, model_config)

    model.classifier.apply(init_weights)

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
    )

    tokenized_train_ds = load_from_disk(f"{train_config.tokenized_ds_dir}/train")
    tokenized_eval_ds = load_from_disk(f"{train_config.tokenized_ds_dir}/eval")

    train_dataloader = torch.utils.data.DataLoader(
        tokenized_train_ds,  # type:ignore
        batch_size=train_config.batch_size,
        collate_fn=data_collator,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        tokenized_eval_ds,  # type:ignore
        batch_size=train_config.batch_size,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
    )

    encoder_params = model.encoder.parameters()
    head_params = model.classifier.parameters()

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": train_config.encoder_lr},
            {"params": head_params, "lr": train_config.head_lr},
        ],
        weight_decay=train_config.weight_decay,
        fused=True,
    )

    total_steps = (
        len(train_dataloader) * train_config.num_epochs
    ) // train_config.gradient_accumulation_steps

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
    criterion = AsymmetricLoss(
        gamma_neg=4.0, gamma_pos=0.0
    )
    model.train()

    # freeze encoder for first 500 steps
    # accelerator.unwrap_model(model).encoder.requires_grad_(False)

    for epoch in range(train_config.num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for step, batch in tqdm(
            enumerate(train_dataloader), total=len(train_dataloader)
        ):
            # if step == 500 and epoch == 0:
            #     accelerator.unwrap_model(model).encoder.requires_grad_(True)
            #     print("Unfreezing encoder weights")

            with accelerator.accumulate(model):
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = criterion(logits, batch["labels"])

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                epoch_loss += loss.detach().item()
                num_batches += 1

            if step % 150 == 0:
                accelerator.print(
                    f"Epoch {epoch + 1} | Step {step} | TRAIN  Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / num_batches
        eval_loss = evaluate_bayesian(
            criterion, model, eval_dataloader, accelerator, num_samples=train_config.num_samples, max_batch_size=max_batch_size
        )
        accelerator.print(
            f"FINISH EPOCH {epoch + 1} | TRAIN Loss: {avg_loss:.4f} | Eval Loss: {eval_loss:.4f}"
        )

        save_emcoder(
            accelerator,
            model,
            f"{save_path}/epoch_{epoch + 1}",
            model_config,
            train_config,
            tokenizer=tokenizer,
            loss=avg_loss,
            eval_loss=eval_loss,
        )

        if accelerator.device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    SAVE_PATH = "models/v1_5/emcoder"
    ENCODER_PATH = "models/v1/pretrain/checkpoints/epoch_2/step_40000"
    N_SAMPLES = 15
    MAX_BATCH_SIZE = 5

    main(
        save_path=SAVE_PATH,
        encoder_path=ENCODER_PATH,
        num_samples=N_SAMPLES,
        max_batch_size=MAX_BATCH_SIZE,
    )
