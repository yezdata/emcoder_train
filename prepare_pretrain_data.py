import os
import shutil
from tqdm import tqdm
from datasets import (
    Dataset,
    Features,
    Value,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from transformers import AutoTokenizer

TEXT_FEATURES = Features({"text": Value(dtype="string")})


def filter_empty_texts(x):
    return x["text"] is not None and len(x["text"].strip()) > 0


def tokenize_batch(x, tokenizer, max_seq_len):
    return tokenizer(
        x["text"],
        add_special_tokens=False,
        padding=False,
        truncation=True,
        max_length=max_seq_len,
    )


# --- OpenWebText ---
def get_owt(target_owt_samples: int):
    full = load_dataset("Skylion007/openwebtext", split="train")
    return full.select(range(target_owt_samples)).select_columns(["text"])


# --- C4 ---
def get_c4(target_c4_samples: int):
    c4_stream = load_dataset("allenai/c4", "en", split="train", streaming=True)
    c4_stream = c4_stream.shuffle(seed=42, buffer_size=1_000_000).take(
        target_c4_samples
    )
    data_list = []
    for x in tqdm(c4_stream, total=target_c4_samples, desc="Building C4 in RAM"):
        text_val = x.get("text", "")
        if text_val and len(text_val.strip()) > 0:
            data_list.append({"text": text_val})

    return Dataset.from_list(data_list, features=TEXT_FEATURES)


# --- Wikipedia ---
def get_wiki(target_wiki_samples: int):
    wiki_full = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    size = min(target_wiki_samples, len(wiki_full))
    return wiki_full.select(range(size)).select_columns(["text"])


def process_and_save_shard(name, get_dataset_fn, tokenizer, max_seq_len, cache_path):
    if os.path.exists(cache_path):
        return load_from_disk(cache_path)

    print(f"[{name}] Generating and processing...")
    ds = get_dataset_fn()

    if name == "C4":
        raw_txt_tmp = f"{cache_path}_raw_text_tmp"
        if os.path.exists(raw_txt_tmp):
            ds = load_from_disk(raw_txt_tmp)
        else:
            print(f"[{name}] flushing C4 raw text to disk...")
            ds.save_to_disk(raw_txt_tmp)
            ds = load_from_disk(raw_txt_tmp)

    if name != "C4":
        print(f"[{name}] Filtering...")
        ds = ds.filter(
            filter_empty_texts,
            num_proc=16,
        )

    tokenized = ds.map(
        tokenize_batch,
        fn_kwargs={"tokenizer": tokenizer, "max_seq_len": max_seq_len},
        batched=True,
        batch_size=4000,
        num_proc=16,
        remove_columns=["text"],
    )

    tmp_cache_path = f"{cache_path}.tmp"
    if os.path.exists(tmp_cache_path):
        shutil.rmtree(tmp_cache_path)

    tokenized.save_to_disk(tmp_cache_path)
    os.rename(tmp_cache_path, cache_path)

    return tokenized


def main():
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    max_seq_len = 512
    num_samples = 12_000_000
    # num_samples = len(openwebtext_full)

    target_owt_samples = int(num_samples * 0.5)
    target_c4_samples = int(num_samples * 0.3)
    target_wiki_samples = int(num_samples * 0.2)

    tmp_dir = "data/tmp_shards"
    os.makedirs(tmp_dir, exist_ok=True)

    owt_ds = process_and_save_shard(
        "OpenWebText",
        lambda: get_owt(target_owt_samples),
        tokenizer,
        max_seq_len,
        f"{tmp_dir}/owt",
    )

    c4_ds = process_and_save_shard(
        "C4", lambda: get_c4(target_c4_samples), tokenizer, max_seq_len, f"{tmp_dir}/c4"
    )

    wiki_ds = process_and_save_shard(
        "Wikipedia",
        lambda: get_wiki(target_wiki_samples),
        tokenizer,
        max_seq_len,
        f"{tmp_dir}/wiki",
    )

    ds = concatenate_datasets([owt_ds, c4_ds, wiki_ds])

    ds = ds.shuffle(seed=100)
    ds = ds.flatten_indices()

    split_ds = ds.train_test_split(test_size=0.01, seed=42)

    data_path = f"data/mixed_ds_v2_seq{max_seq_len}"
    os.makedirs(data_path, exist_ok=True)

    split_ds["train"].save_to_disk(f"{data_path}/train")
    split_ds["test"].save_to_disk(f"{data_path}/eval")


if __name__ == "__main__":
    main()
