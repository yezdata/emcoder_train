import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
import seaborn as sns
from sklearn.metrics import classification_report, multilabel_confusion_matrix, f1_score

from src.config_models import ClassifierConfig


def plot_per_label_scatters(
    mean_probs: np.ndarray,
    epistemic_unc: np.ndarray,
    aleatoric_unc: np.ndarray,
    all_labels: np.ndarray,
    preds: np.ndarray,
    label_names: list,
    output_dir: str,
) -> None:
    os.makedirs(f"{output_dir}/label_scatters", exist_ok=True)

    for i, label_name in enumerate(label_names):
        # Příprava dat pro konkrétní label
        is_correct = preds[:, i] == all_labels[:, i]

        label_df = pd.DataFrame(
            {
                "Probability": mean_probs[:, i],
                "Epistemic": epistemic_unc[:, i],
                "Aleatoric": aleatoric_unc[:, i],
                "Correct": is_correct,
            }
        )

        # Vykreslení obou grafů vedle sebe pro srovnání
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Aleatoric Scatter
        sns.scatterplot(
            data=label_df,
            x="Aleatoric",
            y="Probability",
            hue="Correct",
            palette={True: "#2ecc71", False: "#e74c3c"},
            alpha=0.4,
            ax=axes[0],
        )
        axes[0].set_title(f"Aleatoric Uncertainty - {label_name}")

        # Epistemic Scatter
        sns.scatterplot(
            data=label_df,
            x="Epistemic",
            y="Probability",
            hue="Correct",
            palette={True: "#2ecc71", False: "#e74c3c"},
            alpha=0.4,
            ax=axes[1],
        )
        axes[1].set_title(f"Epistemic Uncertainty - {label_name}")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/label_scatters/{label_name}_scatters.png")
        plt.close()


def load_inference_outputs(input_file: str) -> tuple:
    data = np.load(input_file)
    logits = data["logits"]  # (n_samples, N, num_labels)
    labels = data["labels"]  # (N, L)

    probs = expit(logits)
    probs = np.clip(probs, 1e-8, 1 - 1e-8)

    mean_probs = probs.mean(axis=0)  # (N, num_labels)

    # Predictive Entropy
    total_unc = -(
        mean_probs * np.log(mean_probs) + (1 - mean_probs) * np.log(1 - mean_probs)
    )

    # Expected Entropy
    sample_entropies = -(probs * np.log(probs) + (1 - probs) * np.log(1 - probs))
    aleatoric_unc = sample_entropies.mean(axis=0)

    # Mutual Information
    epistemic_unc = total_unc - aleatoric_unc
    epistemic_unc = np.maximum(epistemic_unc, 0)

    return mean_probs, total_unc, epistemic_unc, aleatoric_unc, labels


def plot_f1_rejection_curve(
    preds_matrix: np.ndarray,
    uncertainty: np.ndarray,
    all_labels: np.ndarray,
    label_names: list,
    output_dir: str,
) -> None:
    rejection_fractions = np.linspace(0.0, 0.5, 20)

    plt.figure(figsize=(12, 8))

    all_f1_scores = {label: [] for label in label_names}
    macro_f1_scores = []

    for frac in rejection_fractions:
        current_step_f1 = []

        for i, label in enumerate(label_names):
            preds = preds_matrix[:, i]
            unc_class = uncertainty[:, i]
            labels_class = all_labels[:, i]

            sorted_indices = np.argsort(unc_class)
            keep_count = int(len(unc_class) * (1.0 - frac))

            if keep_count == 0:
                all_f1_scores[label].append(np.nan)
                continue

            keep_indices = sorted_indices[:keep_count]
            filt_preds = preds[keep_indices]
            filt_labels = labels_class[keep_indices]

            if filt_labels.sum() == 0:
                all_f1_scores[label].append(np.nan)
            else:
                f1 = f1_score(filt_labels, filt_preds, zero_division=0)
                all_f1_scores[label].append(f1)
                current_step_f1.append(f1)

        if current_step_f1:
            macro_f1_scores.append(np.mean(current_step_f1))
        else:
            macro_f1_scores.append(np.nan)

    for label in label_names:
        f1_series = np.array(all_f1_scores[label])
        valid_mask = ~np.isnan(f1_series)
        plt.plot(
            rejection_fractions[: len(f1_series)][valid_mask],
            f1_series[valid_mask],
            alpha=0.15,
            color="gray",
        )

    macro_series = np.array(macro_f1_scores)
    valid_macro_mask = ~np.isnan(macro_series)
    plt.plot(
        rejection_fractions[: len(macro_series)][valid_macro_mask],
        macro_series[valid_macro_mask],
        color="red",
        linewidth=3,
        label="Macro F1",
    )

    plt.xlabel("Rejection Fraction")
    plt.ylabel("F1 Score")
    plt.title("F1-Rejection Curve")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/f1_rejection_curve.png")
    plt.close()


def plot_uncertainty_ridge(
    uncertainty: np.ndarray, label_names: list, output_dir: str
) -> None:
    plt.figure(figsize=(12, 16))

    df = pd.DataFrame(uncertainty, columns=label_names)
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
    plt.title("Distribution of Uncertainty per Emotion")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/unc_violin.png")
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
    mean_probs, total_unc, epistemic_unc, aleatoric_unc, all_labels = (
        load_inference_outputs(input_file)
    )

    with open(f"{model_path}/config.json", "r") as f:
        model_config = ClassifierConfig.model_validate_json(f.read())

    # Calculate metrics
    label_names = [model_config.id2label[i] for i in range(model_config.num_labels)]

    preds = np.zeros_like(all_labels)
    for i, label_name in enumerate(label_names):
        p_thresh = thresholds[label_name]["p"]
        preds[:, i] = (mean_probs[:, i] > p_thresh).astype(int)

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
    labels_flat = all_labels.flatten()

    is_correct = preds_flat == labels_flat

    plot_df = pd.DataFrame(
        {
            "Probability": prob_flat,
            "Epistemic Uncertainty": epistemic_unc.flatten(),
            "Aleatoric Uncertainty": aleatoric_unc.flatten(),
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
    plt.xlabel("Epistemic Uncertainty (Mutual Information)")
    plt.ylabel("Mean Predicted Probability")
    leg = plt.legend(title="Prediction Correct")
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    plt.savefig(f"{output_dir}/epistemic_unc_scatter.png")
    plt.close()

    sns.scatterplot(
        data=plot_df,
        x="Aleatoric Uncertainty",
        y="Probability",
        hue="Correct",
        palette={True: "#2ecc71", False: "#e74c3c"},
        alpha=0.3,
        edgecolor=None,
    )
    plt.xlabel("Aleatoric Uncertainty (Expected Entropy)")
    plt.ylabel("Mean Predicted Probability")
    leg = plt.legend(title="Prediction Correct")
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    plt.savefig(f"{output_dir}/aleatoric_unc_scatter.png")
    plt.close()

    plot_per_label_scatters(
        mean_probs,
        epistemic_unc,
        aleatoric_unc,
        all_labels,
        preds,
        label_names,
        output_dir,
    )

    plot_uncertainty_ridge(total_unc, label_names, output_dir)

    plot_f1_rejection_curve(preds, total_unc, all_labels, label_names, output_dir)

    mcm = multilabel_confusion_matrix(all_labels, preds)
    for i, label in enumerate(label_names):
        sns.heatmap(mcm[i], annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix for {label}")
        plt.savefig(f"{output_dir}/each_label/confusion_matrix_{label}.png")
        plt.close()


if __name__ == "__main__":
    INPUT_FILE = "outputs/v1_5/test_outputs.npz"
    EMCODER_PATH = "models/v1_5/emcoder/epoch_4"
    OUTPUT_PATH = "outputs/v1_5_optim_thresholds"

    main(OUTPUT_PATH, INPUT_FILE, EMCODER_PATH)
