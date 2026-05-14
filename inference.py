import os

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from datasets import load_from_disk
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import DataCollatorWithPadding, RobertaTokenizerFast

from src.config_models import ClassifierConfig
from src.v1.class_head import ClassificationHead
from src.v1.emcoder_core import EmCoderCore


def main(
    tokenized_ds_dir: str,
    emcoder_path: str,
    num_samples: int,
    max_batch_size: int | None,
    output_file: str,
) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    accelerator = Accelerator(mixed_precision="bf16")

    with open(f"{emcoder_path}/config.json", "r") as f:
        model_config = ClassifierConfig.model_validate_json(f.read())

    encoder = EmCoderCore(model_config)
    model = ClassificationHead(encoder, model_config)

    state_dict = load_file(f"{emcoder_path}/model.safetensors")
    model.load_state_dict(state_dict, strict=True)

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
    )

    tokenized_test_ds = load_from_disk(tokenized_ds_dir)

    test_dataloader = torch.utils.data.DataLoader(
        tokenized_test_ds,
        batch_size=128,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    # TEST LOOP - MC DROPOUT INFERENCE
    model.eval()

    all_probs = []
    all_labels = []

    with torch.inference_mode():
        for batch in tqdm(test_dataloader):
            logits_samples = model.mc_forward(
                batch["input_ids"],
                batch["attention_mask"],
                n_samples=num_samples,
                max_batch_size=max_batch_size,
            )

            probs = torch.sigmoid(logits_samples)  # (n_samples, B, 28)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)  # (N, num_labels)
    all_probs = np.concatenate(all_probs, axis=1)  # (n_samples, N, num_labels)

    mean_probs = all_probs.mean(axis=0)  # (N, num_labels)
    epistemic_unc = all_probs.std(axis=0)  # (N, num_labels)

    p = np.clip(mean_probs, 1e-8, 1 - 1e-8)
    binary_entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))  # (N, num_labels)
    aleatoric_unc = np.sum(binary_entropy, axis=-1)  # (N, )

    # export inference outputs
    export_df = pd.DataFrame(
        {
            "mean_probs": list(mean_probs),
            "epistemic_unc": list(epistemic_unc),
            "aleatoric_unc": aleatoric_unc,
            "labels": list(all_labels),
        }
    )
    export_df.to_parquet(output_file)


if __name__ == "__main__":
    TOKENIZED_DS_DIR = "data/goemotions_v1_seq512/eval"
    EMCODER_PATH = "models/v1/emcoder/epoch_10"
    OUTPUT_FILE = "outputs/v1/val_outputs.parquet"
    N_SAMPLES = 50
    MAX_BATCH_SIZE = 5

    main(TOKENIZED_DS_DIR, EMCODER_PATH, N_SAMPLES, MAX_BATCH_SIZE, OUTPUT_FILE)
