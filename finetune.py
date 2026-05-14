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
from src.types_shapes import ClassLabels, ClassLogits
from src.utils import init_weights, save_emcoder
from src.v1.class_head import ClassificationHead
from src.v1.emcoder_core import EmCoderCore


# TODO: nejde misto .compute_loss primo forward metoda?
# weighted BCE with logits loss
class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, device, pos_weights=None):
        super().__init__()
        self.pos_weights = pos_weights.to(device) if pos_weights is not None else None
        self.loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)

    def compute_loss(self, logits: ClassLogits, labels: ClassLabels) -> torch.Tensor:
        return self.loss_fct(logits, labels.to(logits.dtype))

    def compute_bayesian_loss(
        self, logits_samples: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        n_samples, B, n_labels = logits_samples.shape

        labels_expanded = labels.unsqueeze(0).expand(
            n_samples, -1, -1
        )  # (n_samples, B, num_labels)

        flat_logits = logits_samples.view(-1, n_labels)  # (n_samples * B, n_labels)
        flat_labels = labels_expanded.reshape(-1, n_labels)  # (n_samples * B, n_labels)

        loss = self.loss_fct(flat_logits, flat_labels.to(flat_logits.dtype))

        return loss


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    preds = (probs > 0.5).astype(int)

    return {
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "roc_auc": roc_auc_score(labels, probs, average="macro"),
    }


def evaluate_bayesian(
    criterion: WeightedBCEWithLogitsLoss,
    model: ClassificationHead,
    dataloader: torch.utils.data.DataLoader,
    num_samples: int,
) -> float:
    model.eval()
    total_loss = 0.0
    all_pred_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            logits_samples = model.mc_forward(
                batch["input_ids"], batch["attention_mask"], n_samples=num_samples
            )

            loss = criterion.compute_bayesian_loss(logits_samples, batch["labels"])

            probs = torch.sigmoid(logits_samples)
            mean_probs = probs.mean(dim=0)  # (B, num_labels)

            all_pred_probs.append(mean_probs.cpu())
            all_labels.append(batch["labels"].cpu())
            total_loss += loss.item()

    y_probs = torch.cat(all_pred_probs).numpy()
    y_true = torch.cat(all_labels).numpy()

    print(compute_metrics(y_probs, y_true))

    model.train()
    return total_loss / len(dataloader)


def main(
    save_path: str,
    encoder_path: str,
    num_samples: int,
    bayesian_train: bool = False,
    loss_weights: Literal["sqrt", "log"] | None = None,
) -> None:
    encoder_lr = 1e-5 if bayesian_train else 2e-5
    head_lr = 5e-4 if bayesian_train else 1e-3
    # TODO if model give similar results for all samples, increase model_config.dropout to 0.15 / 0.2
    dropout_rate = 0.15 if bayesian_train else 0.1

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
        bayesian_train=bayesian_train,
        num_samples=num_samples,
        loss_weights=loss_weights,
        tokenized_ds_dir=f"data/goemotions_v1_seq{model_config.max_seq_len}",
        encoder_lr=encoder_lr,
        head_lr=head_lr,
        batch_size=32,
        gradient_accumulation_steps=8,
        num_epochs=10,
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
        num_workers=8,
        pin_memory=True,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        tokenized_eval_ds,  # type:ignore
        batch_size=train_config.batch_size,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=8,
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

    # computing the weights of labels
    pos_counts = np.sum(np.array(tokenized_train_ds["labels"]), axis=0)
    neg_counts = len(tokenized_train_ds) - pos_counts

    # NOTE: USE SQRT / LOG FOR NUM STABILITY
    # - 1e-5 avoids division by zero
    if train_config.loss_weights == "sqrt":
        raw_weights = np.sqrt((neg_counts + 1e-5) / (pos_counts + 1e-5))
    elif train_config.loss_weights == "log":
        raw_weights = 1 + np.log((neg_counts + 1e-5) / (pos_counts + 1e-5))
    else:
        raw_weights = (neg_counts + 1e-5) / (pos_counts + 1e-5)

    # PREVENT EXTREME WEIGTHS
    raw_weights = np.clip(raw_weights, a_min=0.1, a_max=20.0)
    weights = torch.tensor(raw_weights, dtype=torch.float32)

    print("Label Weights:", weights)

    # TRAIN LOOP
    criterion = WeightedBCEWithLogitsLoss(
        device=accelerator.device, pos_weights=weights
    )
    model.train()

    # freeze encoder for first 500 steps
    accelerator.unwrap_model(model).encoder.requires_grad_(False)

    for epoch in range(train_config.num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for step, batch in tqdm(
            enumerate(train_dataloader), total=len(train_dataloader)
        ):
            if step == 500 and epoch == 0:
                accelerator.unwrap_model(model).encoder.requires_grad_(True)
                print("Unfreezing encoder weights")

            with accelerator.accumulate(model):
                if train_config.bayesian_train:
                    logits_samples = model.mc_forward(
                        batch["input_ids"],
                        batch["attention_mask"],
                        n_samples=train_config.num_samples,
                    )

                    if step % 150 == 0:
                        accelerator.print(
                            f"Step {step} | Avg Logits: {logits_samples.mean().item():.4f}"
                        )

                    loss = criterion.compute_bayesian_loss(
                        logits_samples, batch["labels"]
                    )

                else:
                    logits = model(batch["input_ids"], batch["attention_mask"])
                    loss = criterion.compute_loss(logits, batch["labels"])

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                num_batches += 1

            if step % 150 == 0:
                accelerator.print(
                    f"Epoch {epoch + 1} | Step {step} | TRAIN  Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / num_batches
        eval_loss = evaluate_bayesian(
            criterion, model, eval_dataloader, num_samples=train_config.num_samples
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


if __name__ == "__main__":
    SAVE_PATH = "models/v1/emcoder"
    ENCODER_PATH = "models/v1/pretrain/checkpoints/epoch_2/step_40000"
    BAYESIAN_TRAIN = True
    N_SAMPLES = 20
    LOSS_WEIGHTS = "log"

    main(
        save_path=SAVE_PATH,
        encoder_path=ENCODER_PATH,
        num_samples=N_SAMPLES,
        bayesian_train=BAYESIAN_TRAIN,
        loss_weights=LOSS_WEIGHTS,
    )
