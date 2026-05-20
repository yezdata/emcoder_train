import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_from_disk
from dotenv import load_dotenv
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,
)

from src.config_models import EmCoderConfig, EmCoderEncoderConfig, FinetuneConfig
from src.losses import AsymmetricLoss, WeightedBCEWithLogitsLoss
from src.utils import init_weights, save_emcoder
from src.v2.emcoder import EmCoder


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    preds = (probs > 0.5).astype(int)

    return {
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),  # type:ignore
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),  # type:ignore
        "roc_auc": roc_auc_score(labels, probs, average="macro"),
    }


def evaluate_bayesian(
    criterion: AsymmetricLoss | WeightedBCEWithLogitsLoss,
    model: EmCoder,
    dataloader: torch.utils.data.DataLoader,
    n_samples: int,
    max_batch_size: int,
) -> float:
    model.eval()

    total_loss = 0.0
    all_pred_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc="Evaluating"):
            logits_samples = model.mc_forward(
                batch["input_ids"],
                batch["attention_mask"],
                n_samples=n_samples,
                max_batch_size=max_batch_size,
            )
            logits_samples = logits_samples.detach()

            loss = criterion.compute_bayesian_loss(logits_samples, batch["labels"])
            total_loss += loss.detach().item()

            probs = logits_samples.sigmoid()
            mean_probs = probs.mean(dim=0)  # (B, num_labels)

            all_pred_probs.append(mean_probs.detach().cpu().numpy())
            all_labels.append(batch["labels"].detach().cpu().numpy())

    y_probs = np.concatenate(all_pred_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    print(compute_metrics(y_probs, y_true))

    return total_loss / len(dataloader)


def main(
    save_path: str,
    encoder_path: str,
) -> None:
    os.makedirs(save_path, exist_ok=True)

    load_dotenv()

    set_seed(1337)

    train_config = FinetuneConfig()  # type:ignore
    max_batch_size = train_config.n_samples

    with open(f"{encoder_path}/config.json", "r") as f:
        encoder_config = EmCoderEncoderConfig.model_validate_json(f.read())

    with open("data/goemotions/emotions.txt", "r", encoding="utf-8") as f:
        label_names = [line.strip() for line in f.readlines()]

    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in enumerate(label_names)}

    model_config = EmCoderConfig(
        **encoder_config.model_dump(),
        base_encoder_path=encoder_path,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
    )

    model = EmCoder(model_config)

    model.encoder.load_state_dict(
        torch.load(f"{encoder_path}/encoder_only.pth", map_location=accelerator.device),
        strict=True,
    )

    model.classifier.apply(init_weights)
    # Scale down the final logit layer -> uniform outputs for classes
    nn.init.trunc_normal_(model.classifier[-1].weight, std=0.005)
    if model.classifier[-1].bias is not None:
        nn.init.zeros_(model.classifier[-1].bias)

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

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

    # computing the weights of labels
    pos_counts = np.sum(np.array(tokenized_train_ds["labels"]), axis=0)
    neg_counts = len(tokenized_train_ds) - pos_counts

    criterion = WeightedBCEWithLogitsLoss(pos_counts=pos_counts, neg_counts=neg_counts)
    print("Label Weights:", criterion.pos_weights)

    model, optimizer, train_dataloader, eval_dataloader, criterion = (
        accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, criterion
        )
    )

    total_steps = (
        len(train_dataloader) * train_config.num_epochs
    ) // train_config.gradient_accumulation_steps

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * train_config.lr_warmup),
        num_training_steps=total_steps,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # TRAIN LOOP

    # freeze encoder for first 500 steps
    # accelerator.unwrap_model(model).encoder.requires_grad_(False)

    for epoch in range(train_config.num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        model.train()
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
                    f"Epoch {epoch + 1} | Step {step} | TRAIN  Loss: {epoch_loss / num_batches:.4f}"
                )

        avg_loss = epoch_loss / num_batches

        eval_loss = evaluate_bayesian(
            criterion,
            model,
            eval_dataloader,
            n_samples=train_config.n_samples,
            max_batch_size=max_batch_size,
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
            loss=avg_loss,
            eval_loss=eval_loss,
        )


if __name__ == "__main__":
    SAVE_PATH = "models/v1_5/emcoder"
    ENCODER_PATH = "models/v1/pretrain/checkpoints/epoch_1/step_120000"

    main(
        save_path=SAVE_PATH,
        encoder_path=ENCODER_PATH,
    )
