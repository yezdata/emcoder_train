import os

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_from_disk
from safetensors.torch import load_model
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding

from src.config_models import EmCoderConfig, InferenceConfig
from src.v2.emcoder import EmCoder


def main() -> None:
    inference_config = InferenceConfig()  # type:ignore

    # max_batch_size = inference_config.n_samples
    max_batch_size = 5

    os.makedirs(os.path.dirname(inference_config.output_file), exist_ok=True)

    accelerator = Accelerator()

    with open(f"{inference_config.emcoder_path}/config.json", "r") as f:
        model_config = EmCoderConfig.model_validate_json(f.read())

    model = EmCoder(model_config)

    load_model(model, f"{inference_config.emcoder_path}/model.safetensors")

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
    )

    tokenized_test_ds = load_from_disk(inference_config.tokenized_ds_dir)

    test_dataloader = torch.utils.data.DataLoader(
        tokenized_test_ds,  # type:ignore
        batch_size=inference_config.batch_size,
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

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            logits_samples = model.mc_forward(
                batch["input_ids"],
                batch["attention_mask"],
                n_samples=inference_config.n_samples,
                max_batch_size=max_batch_size,
            )
            logits_samples = logits_samples.detach()

            all_logits.append(logits_samples.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=1)  # (n_samples, N, num_labels)
    all_labels = np.concatenate(all_labels, axis=0)  # (N, num_labels)

    # export inference outputs
    np.savez_compressed(
        inference_config.output_file, logits=all_logits, labels=all_labels
    )


if __name__ == "__main__":
    main()
