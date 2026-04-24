import pandas as pd


with open("data/goemotions/emotions.txt", "r", encoding="utf-8") as f:
    label_names = [line.strip() for line in f.readlines()]


def to_multi_hot(label_indices, num_labels=28):
    vector = [0.0] * num_labels
    for idx in label_indices:
        vector[int(idx)] = 1.0
    return vector


train_df = pd.read_csv("data/goemotions/raw/train.tsv", names=["text", "labels", "id"], sep="\t")
val_df = pd.read_csv("data/goemotions/raw/dev.tsv", names=["text", "labels", "id"], sep="\t")
test_df = pd.read_csv("data/goemotions/raw/test.tsv", names=["text", "labels", "id"], sep="\t")



for df in [train_df, val_df, test_df]:
    print(f"DF COUNT: {df.shape[0]}")
    print(f"LONGEST TEXT: {df['text'].str.len().max()}")

    df["labels"] = df["labels"].map(lambda x: [int(i) for i in x.split(",")])
    df["labels"] = df["labels"].map(lambda x: to_multi_hot(x, len(label_names)))


train_df[["text", "labels"]].to_json(
    "data/goemotions/train_data.jsonl", orient="records", lines=True, force_ascii=False
)
val_df[["text", "labels"]].to_json(
    "data/goemotions/val_data.jsonl", orient="records", lines=True, force_ascii=False
)
test_df[["text", "labels"]].to_json(
    "data/goemotions/test_data.jsonl", orient="records", lines=True, force_ascii=False
)