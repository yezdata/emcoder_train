from datasets import load_dataset
from transformers import RobertaTokenizerFast


def main():
    ds = load_dataset("Skylion007/openwebtext", split="train")
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    max_seq_len = 514

    tokenized_ds = ds.map(
        lambda x: tokenizer(x["text"], padding=False, truncation=True, max_length=max_seq_len),
        batched=True,
        num_proc=16,
        remove_columns=ds.column_names
    )
    split_ds = tokenized_ds.train_test_split(test_size=0.01)

    train_ds = split_ds["train"]
    eval_ds = split_ds["test"]

    data_path = f"data/openwebtext_v1_seq{max_seq_len}"
    train_ds.save_to_disk(f"{data_path}/train")
    eval_ds.save_to_disk(f"{data_path}/eval")

if __name__ == "__main__":
    main()