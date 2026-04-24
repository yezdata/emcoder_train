from datasets import load_dataset
from transformers import RobertaTokenizerFast
import numpy as np



def main():
    ds = load_dataset("json", data_files={"train": "data/goemotions/train_data.jsonl", "eval": "data/goemotions/val_data.jsonl", "test": "data/goemotions/test_data.jsonl"})


    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    max_seq_len = 512

    tokenized_ds = ds.map(
        lambda x: tokenizer(x["text"], padding=False, truncation=True, max_length=max_seq_len),
        batched=True,
        num_proc=16,
        remove_columns=["text"]
    )
    
    tokenized_ds.set_format("torch")
    
    tokenized_ds.save_to_disk(f"data/goemotions_v1_seq{max_seq_len}")

if __name__ == "__main__":
    main()