import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, multilabel_confusion_matrix

from src.config_models import ClassifierConfig


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
    output_dir: str,
    input_file: str,
    model_path: str,
) -> None:
    os.makedirs(f"{output_dir}/each_label", exist_ok=True)

    # load optimized thresholds
    with open(f"{model_path}/thresholds.json", "r") as f:
        thresholds = json.load(f)

    plt.rcParams["figure.figsize"] = (14, 10)
    sns.set_theme(style="whitegrid")

    # load inference outputs
    df = pd.read_parquet(input_file)

    mean_probs = np.stack(df["mean_probs"].values)
    epistemic_unc = np.stack(df["epistemic_unc"].values)
    aleatoric_unc = df["aleatoric_unc"].values
    all_labels = np.stack(df["labels"].values)

    with open(f"{model_path}/config.json", "r") as f:
        model_config = ClassifierConfig.model_validate_json(f.read())

    # Calculate metrics
    label_names = [model_config.id2label[i] for i in range(model_config.num_labels)]

    preds = np.zeros_like(all_labels)
    for i, label_name in enumerate(label_names):
        p_thresh = thresholds[label_name]["p"]
        u_thresh = thresholds[label_name]["u"]
        preds[:, i] = (
            (mean_probs[:, i] > p_thresh) & (epistemic_unc[:, i] < u_thresh)
        ).astype(int)

    report_dict = classification_report(
        all_labels, preds, target_names=label_names, zero_division=0, output_dict=True
    )
    df_report = pd.DataFrame(report_dict).transpose()
    df_report = df_report.round(3)
    df_report["support"] = df_report["support"].astype(int)
    df_report.index.name = "label"
    df_report = df_report.reset_index()
    df_report.to_csv(f"{output_dir}/classification_report.csv", index=False)
    print(df_report)

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
    preds_flat = preds.flatten()
    unc_flat = epistemic_unc.flatten()
    labels_flat = all_labels.flatten()

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
        alpha=0.3,
        edgecolor=None,
    )
    plt.xlabel("Epistemic Uncertainty (Std Dev)")
    plt.ylabel("Mean Predicted Probability")
    leg = plt.legend(title="Prediction Correct")
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
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
    INPUT_FILE = "outputs/v1/inference_outputs.parquet"
    EMCODER_PATH = "models/v1/emcoder/epoch_10"
    OUTPUT_PATH = "outputs/v1_optim_thresholds"

    main(OUTPUT_PATH, INPUT_FILE, EMCODER_PATH)
