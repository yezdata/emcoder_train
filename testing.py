import torch
import numpy as np
from safetensors.torch import load_file
from datasets import load_from_disk
from transformers import RobertaTokenizerFast, DataCollatorWithPadding
from accelerate import Accelerator
from tqdm import tqdm
import os
import pandas as pd
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


from src.config_models import ClassifierConfig
from src.v1.class_head import ClassificationHead
from src.v1.emcoder_core import EmCoderCore


def plot_uncertainty_ridge(
    epistemic_unc: np.ndarray, label_names: list, output_dir: str
) -> None:
    plt.figure(figsize=(12, 16))

    df = pd.DataFrame(epistemic_unc, columns=label_names)
    df_melted = df.melt(var_name="Emotion", value_name="Uncertainty")

    sns.violinplot(
        data=df_melted,
        x="Uncertainty",
        y="Emotion",
        orient="h",
        cut=0,
        scale="width",
        inner="quartile",
    )
    plt.title("Distribution of Epistemic Uncertainty per Emotion")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/epistemic_unc_violin.png")
    plt.close()


def main(
    tokenized_ds_dir: str, emcoder_path: str, n_samples: int, output_dir: str
) -> None:
    os.makedirs(f"{output_dir}/each_label", exist_ok=True)

    plt.rcParams["figure.figsize"] = (14, 10)
    sns.set_theme(style="whitegrid")

    accelerator = Accelerator(mixed_precision="bf16")

    with open(f"{emcoder_path}/model_config.json", "r") as f:
        model_config = ClassifierConfig.model_validate_json(f.read())

    encoder = EmCoderCore(model_config)
    model = ClassificationHead(encoder, model_config)

    state_dict = load_file(f"{emcoder_path}/model.safetensors")
    model.load_state_dict(state_dict, strict=True)

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
    )

    tokenized_test_ds = load_from_disk(f"{tokenized_ds_dir}/test")

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
                batch["input_ids"], batch["attention_mask"], n_samples=n_samples
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

    label_names = [model_config.id2label[i] for i in range(model_config.num_labels)]

    # export inference outputs
    export_df = pd.DataFrame(
        {
            "mean_probs": list(mean_probs),
            "epistemic_unc": list(epistemic_unc),
            "labels": list(all_labels),
            "aleatoric_unc": aleatoric_unc,
        }
    )
    export_df.to_parquet(f"{output_dir}/inference_outputs.parquet")

    # Calculate metrics

    # TODO
    # epistemic uncertainty threshold
    # threshold = 0.15

    # max_epistemic = np.max(epistemic_unc, axis=-1)
    # is_uncertain = max_epistemic > threshold

    # metrics_certain = compute_metrics(mean_probs[~is_uncertain], all_labels[~is_uncertain])
    # metrics_uncertain = compute_metrics(mean_probs[is_uncertain], all_labels[is_uncertain])

    # print(f"Predictions with high confidence (N={np.sum(~is_uncertain)}): F1 Micro = {metrics_certain['f1_micro']}")
    # print(f"Predictions with low confidence (N={np.sum(is_uncertain)}): F1 Micro = {metrics_uncertain['f1_micro']}")

    # TODO find best threshold
    preds = (mean_probs > 0.5).astype(int)

    report_dict = classification_report(
        all_labels, preds, target_names=label_names, zero_division=0, output_dict=True
    )
    df = pd.DataFrame(report_dict).transpose().round(3)
    df["support"] = df["support"].astype(int)
    df.to_csv(f"{output_dir}/classification_report.csv", index=False)
    print(df)

    conf_matrix = np.dot(all_labels.T, preds)

    # normalization to recall -> rows
    row_sums = all_labels.sum(axis=0, keepdims=True).T
    conf_matrix_pct = (conf_matrix / (row_sums + 1e-8)) * 100

    plt.figure(figsize=(15, 12))
    sns.heatmap(
        conf_matrix_pct,
        annot=False,
        cmap="GnBu",
        xticklabels=label_names,
        yticklabels=label_names,
    )
    plt.title(
        "Multi-label Confusion Matrix (% of times label j predicted when i is true)"
    )
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()

    prob_flat = mean_probs.flatten()
    unc_flat = epistemic_unc.flatten()
    labels_flat = all_labels.flatten()
    preds_flat = (prob_flat > 0.5).astype(int)

    is_correct = preds_flat == labels_flat

    plot_df = pd.DataFrame(
        {
            "Probability": prob_flat,
            "Epistemic Uncertainty": unc_flat,
            "Correct": is_correct,
        }
    )

    sns.scatterplot(
        data=plot_df,
        x="Epistemic Uncertainty",
        y="Probability",
        hue="Correct",
        palette={True: "#2ecc71", False: "#e74c3c"},
        alpha=0.4,
        edgecolor=None,
    )
    plt.xlabel("Epistemic Uncertainty")
    plt.ylabel("Mean Predicted Probability")
    plt.legend(title="Prediction Correct", labels=["Incorrect", "Correct"])
    plt.savefig(f"{output_dir}/epistemic_unc_scatter.png")
    plt.close()

    sns.scatterplot(x=aleatoric_unc.flatten(), y=mean_probs.max(axis=-1), alpha=0.5)
    plt.xlabel("Aleatoric Uncertainty")
    plt.ylabel("Max Mean Predicted Probability")
    plt.savefig(f"{output_dir}/aleatoric_unc_scatter.png")
    plt.close()

    plot_uncertainty_ridge(epistemic_unc, label_names, output_dir)

    mcm = multilabel_confusion_matrix(all_labels, preds)
    for i, label in enumerate(label_names):
        sns.heatmap(mcm[i], annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix for {label}")
        plt.savefig(f"{output_dir}/each_label/confusion_matrix_{label}.png")
        plt.close()


if __name__ == "__main__":
    TOKENIZED_DS_DIR = "data/goemotions_v1_seq512"
    EMCODER_PATH = "models/v1/emcoder/epoch_10"
    OUTPUT_DIR = "outputs/v1"
    N_SAMPLES = 50

    main(TOKENIZED_DS_DIR, EMCODER_PATH, N_SAMPLES, OUTPUT_DIR)
