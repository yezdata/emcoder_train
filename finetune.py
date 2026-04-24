import torch
import torch.nn as nn
import numpy as np
from accelerate import Accelerator
from transformers import DataCollatorWithPadding, RobertaTokenizerFast, get_cosine_schedule_with_warmup
from datasets import load_from_disk
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score

from src.types_shapes import ClassLabels, ClassLogits
from src.utils import save_emocoder
from src.v1.emocoder import EmoCoderCore
from src.v1.class_head import ClassificationHead
from config import ClassifierConfig, ModelConfig, FinetuneConfig


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




# weighted BCE with logits loss 
class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, device, pos_weights=None):
        super().__init__()
        self.pos_weights = pos_weights.to(device) if pos_weights is not None else None
        self.loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)


    def compute_loss(self, logits: ClassLogits, labels: ClassLabels) -> torch.Tensor:
        return self.loss_fct(logits, labels.to(logits.dtype))


    def compute_bayesian_loss(self, logits_samples: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        n_samples, B, n_labels = logits_samples.shape

        labels_expanded = labels.unsqueeze(0).expand(n_samples, -1, -1) # (n_samples, B, num_labels)

        flat_logits = logits_samples.view(-1, n_labels) # (n_samples * B, n_labels)
        flat_labels = labels_expanded.reshape(-1, n_labels) # (n_samples * B, n_labels)
        
        loss = self.loss_fct(flat_logits, flat_labels.to(flat_logits.dtype))
        
        return loss


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    preds = (probs > 0.3).astype(int)

    return {
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "roc_auc": roc_auc_score(labels, probs, average="macro", multi_class="ovr"),
    }


def evaluate_bayesian(criterion: WeightedBCEWithLogitsLoss, model: ClassificationHead, dataloader: torch.utils.data.DataLoader) -> float:
    model.eval()
    total_loss = 0.0
    all_pred_probs = []
    all_labels = []
    for batch in dataloader:
        with torch.no_grad():
            logits_samples = model.mc_forward(batch["input_ids"], batch["attention_mask"], n_samples=5)

            loss = criterion.compute_bayesian_loss(logits_samples, batch["labels"])

            probs = torch.sigmoid(logits_samples)
            mean_probs = probs.mean(dim=0) # (B, num_labels)


        all_pred_probs.append(mean_probs.cpu())
        all_labels.append(batch["labels"].cpu())
        total_loss += loss.item()

    y_probs = torch.cat(all_pred_probs).numpy()
    y_true = torch.cat(all_labels).numpy()

    print(compute_metrics(y_probs, y_true))
        
    model.train()
    return total_loss / len(dataloader)



def main(save_path: str, encoder_path: str, bayesian_train: bool = False) -> None:
    encoder_lr = 1e-5 if bayesian_train else 2e-5
    head_lr = 5e-5 if bayesian_train else 1e-3
    # TODO, pokud model dava skoro stejne vysledky pro vsechny samples, zvednout model_config.dropout na 0.15 / 0.2
    dropout_rate = 0.15 if bayesian_train else 0.1


    with open(f"{encoder_path}/model_config.json", "r") as f:
        encoder_config = ModelConfig.model_validate_json(f.read())

    
    with open("data/goemotions/emotions.txt", "r", encoding="utf-8") as f:
        label_names = [line.strip() for line in f.readlines()]

    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in enumerate(label_names)}

    model_config = ClassifierConfig(**encoder_config.model_dump(), base_encoder_path=encoder_path, num_labels=len(label_names), id2label=id2label, label2id=label2id)
    model_config.dropout = dropout_rate


    train_config = FinetuneConfig(tokenized_ds_dir=f"data/goemotions_v1_seq{model_config.max_seq_len}", encoder_lr=encoder_lr, head_lr=head_lr, batch_size=32, gradient_accumulation_steps=8, num_epochs=10, lr_warmup=0.1, weight_decay=0.01)


    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=train_config.gradient_accumulation_steps)


    encoder = EmoCoderCore(encoder_config)
    encoder.load_state_dict(torch.load(f"{encoder_path}/encoder_only.pth"), strict=True)


    model = ClassificationHead(encoder, model_config, model_config.num_labels)

    model.classifier.apply(init_weights)


    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")


    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, 
    )

    tokenized_train_ds = load_from_disk(f"{train_config.tokenized_ds_dir}/train")
    tokenized_eval_ds = load_from_disk(f"{train_config.tokenized_ds_dir}/eval")

    train_dataloader = torch.utils.data.DataLoader(tokenized_train_ds, batch_size=train_config.batch_size, collate_fn=data_collator, shuffle=True, num_workers=8, pin_memory=True)
    eval_dataloader = torch.utils.data.DataLoader(tokenized_eval_ds, batch_size=train_config.batch_size, collate_fn=data_collator, shuffle=False, num_workers=8, pin_memory=True)



    encoder_params = model.encoder.parameters()
    head_params = model.classifier.parameters()

    optimizer = torch.optim.AdamW(
        [
            {'params': encoder_params, 'lr': train_config.encoder_lr},
            {'params': head_params, 'lr': train_config.head_lr}
        ],
        weight_decay=train_config.weight_decay,
        fused=True,
    )

    total_steps = (len(train_dataloader) * train_config.num_epochs) // train_config.gradient_accumulation_steps

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * train_config.lr_warmup),
        num_training_steps=total_steps
    )


    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )


    # computing the weights of labels
    pos_counts = np.sum(np.array(tokenized_train_ds["labels"]), axis=0)
    neg_counts = len(tokenized_train_ds) - pos_counts

    # NOTE: USE SQRT FOR NUM STABILITY
    # - 1e-5 avoid division by zero
    raw_weights = np.sqrt((neg_counts + 1e-5) / (pos_counts + 1e-5))
    # PREVENT EXTREME WEIGTHS
    raw_weights = np.clip(raw_weights, a_min=0.1, a_max=50.0)
    weights = torch.tensor(raw_weights, dtype=torch.float32)
    # NORMALIZATION (mean=1)
    weights = weights / weights.mean()
    print("Label Weights:", weights)





    # TRAIN LOOP
    criterion = WeightedBCEWithLogitsLoss(device=accelerator.device, pos_weights=weights)
    model.train()

    for epoch in range(train_config.num_epochs):
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            with accelerator.accumulate(model):
                if bayesian_train:
                    logits_samples = model.mc_forward(batch["input_ids"], batch["attention_mask"], n_samples=5)

                    if step % 50 == 0:
                        avg_logits = logits_samples.mean().item()
                        accelerator.print(f"Step {step} | Avg Logits: {avg_logits:.4f}")

                    loss = criterion.compute_bayesian_loss(logits_samples, batch["labels"])

                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                else:
                    outputs = model(batch["input_ids"], batch["attention_mask"])
                    loss = criterion.compute_loss(outputs, batch["labels"])

                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
            
            if step % 150 == 0:
                accelerator.print(f"Epoch {epoch + 1} | Step {step} | TRAIN  Loss: {loss.item():.4f}")


        eval_loss = evaluate_bayesian(criterion, model, eval_dataloader)
        accelerator.print(f"FINISH EPOCH {epoch + 1} | TRAIN Loss: {loss.item():.4f} | Eval Loss: {eval_loss:.4f}")

        save_emocoder(accelerator, model, f"{save_path}/epoch_{epoch + 1}", model_config, train_config, tokenizer=tokenizer, loss=loss.item(), eval_loss=eval_loss)


             




if __name__ == "__main__":
    SAVE_PATH = "models/v1/emocoder"
    ENCODER_PATH = "models/v1/pretrain/checkpoints/epoch_2/step_40000"
    BAYESIAN_TRAIN = False


    main(SAVE_PATH, ENCODER_PATH, BAYESIAN_TRAIN)