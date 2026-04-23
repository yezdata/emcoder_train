from datasets import load_dataset
from transformers import RobertaTokenizerFast
import numpy as np


def encode_labels(example):
    one_hot = np.zeros(28, dtype=np.float32)
    for label in example["labels"]:
        one_hot[label] = 1.0
    return {"target_labels": one_hot}

def main():
    ds = load_dataset("mrm8488/goemotions")
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    max_seq_len = 512

    tokenized_ds = ds.map(
        lambda x: tokenizer(x["text"], padding=False, truncation=True, max_length=max_seq_len),
        batched=True,
        num_proc=16,
        remove_columns=ds.column_names
    )
    split_ds = tokenized_ds.train_test_split(test_size=0.15)
    
    tokenized = split_ds.map(encode_labels)
    
    # Odstranění nepotřebných sloupců, necháme jen input_ids, attention_mask a target_labels
    tokenized = tokenized.remove_columns(["text", "labels", "id"])
    tokenized.set_format("torch")
    
    tokenized.save_to_disk("data/goemotions_v1_processed")

if __name__ == "__main__":
    main()