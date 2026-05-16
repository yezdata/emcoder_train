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
        batch_size=64,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
    )

    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    # TEST LOOP - MC DROPOUT INFERENCE
    model.eval()

    all_logits = []
    all_labels = []

    with torch.inference_mode():
        for batch in tqdm(test_dataloader):
            logits_samples = model.mc_forward(
                batch["input_ids"],
                batch["attention_mask"],
                n_samples=num_samples,
                max_batch_size=max_batch_size,
            )

            all_logits.append(logits_samples.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=1)  # (n_samples, N, num_labels)
    all_labels = np.concatenate(all_labels, axis=0)  # (N, num_labels)

    # export inference outputs
    np.savez_compressed(output_file, logits=all_logits, labels=all_labels)


if __name__ == "__main__":
    TOKENIZED_DS_DIR = "data/goemotions_v1_seq512/test"
    EMCODER_PATH = "models/v1_5/emcoder/epoch_4"
    OUTPUT_FILE = "outputs/v1_5/test_outputs.npz"
    N_SAMPLES = 50
    MAX_BATCH_SIZE = 5

    main(TOKENIZED_DS_DIR, EMCODER_PATH, N_SAMPLES, MAX_BATCH_SIZE, OUTPUT_FILE)
