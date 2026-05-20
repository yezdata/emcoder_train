import os

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    max_seq_len = 512

    openwebtext_full = load_dataset("Skylion007/openwebtext", split="train")
    num_samples = 12_000_000
    # num_samples = len(openwebtext_full)

    # Data Mixture
    target_owt_samples = int(num_samples * 0.5)
    target_c4_samples = int(num_samples * 0.3)
    target_wiki_samples = int(num_samples * 0.2)

    # OpenWebText
    owt_subset = openwebtext_full.shuffle(seed=42).select(range(target_owt_samples))
    owt_subset = owt_subset.select_columns(["text"])

    # C4
    c4_stream = load_dataset("allenai/c4", "en", split="train", streaming=True)
    c4_stream = c4_stream.shuffle(seed=42).take(target_c4_samples)
    c4_subset = Dataset.from_generator(
        lambda: ({"text": x["text"]} for x in c4_stream), features=owt_subset.features
    )

    # Wikipedia
    wiki_full = load_dataset("wikipedia", "20231101.en", split="train")
    wiki_subset = wiki_full.shuffle(seed=42).select(
        range(min(target_wiki_samples, len(wiki_full)))
    )
    wiki_subset = wiki_subset.select_columns(["text"])

    ds = concatenate_datasets([owt_subset, c4_subset, wiki_subset])
    ds = ds.filter(lambda x: x["text"] is not None and len(x["text"].strip()) > 0)
    ds = ds.shuffle(seed=100)

    tokenized_ds = ds.map(
        lambda x: tokenizer(
            x["text"],
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=max_seq_len,
        ),
        batched=True,
        batch_size=4000,
        num_proc=16,
        remove_columns=["text"],
    )

    split_ds = tokenized_ds.train_test_split(test_size=0.01, seed=42)

    train_ds = split_ds["train"]
    eval_ds = split_ds["test"]

    data_path = f"data/mixed_ds_v2_seq{max_seq_len}"
    os.makedirs(data_path, exist_ok=True)

    train_ds.save_to_disk(f"{data_path}/train")
    eval_ds.save_to_disk(f"{data_path}/eval")


if __name__ == "__main__":
    main()
