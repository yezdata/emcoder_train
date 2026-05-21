import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import expit
from sklearn.calibration import calibration_curve
from sklearn.metrics import classification_report, f1_score, multilabel_confusion_matrix

from src.config_models import EmCoderConfig, EvalConfig


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
        is_correct = preds[:, i] == all_labels[:, i]

        label_df = pd.DataFrame(
            {
                "Probability": mean_probs[:, i],
                "Epistemic": epistemic_unc[:, i],
                "Aleatoric": aleatoric_unc[:, i],
                "Correct": is_correct,
            }
        )

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

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


def plot_per_label_calibrations(
    all_labels: np.ndarray, mean_probs: np.ndarray, label_names: list, output_dir: str
) -> None:
    os.makedirs(f"{output_dir}/label_calibrations", exist_ok=True)

    for i, label in enumerate(label_names):
        prob_true, prob_pred = calibration_curve(
            all_labels[:, i], mean_probs[:, i], n_bins=10, strategy="quantile"
        )
        plt.plot(
            [0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated"
        )
        plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label="Model")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of True Labels")
        plt.title(f"Calibration Curve for {label}")
        plt.legend(loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(f"{output_dir}/label_calibrations/calibration_{label}.png")
        plt.close()


def plot_per_label_cms(
    all_labels: np.ndarray, preds: np.ndarray, label_names: list, output_dir: str
) -> None:
    os.makedirs(f"{output_dir}/label_cms", exist_ok=True)

    mcm = multilabel_confusion_matrix(all_labels, preds)
    for i, label in enumerate(label_names):
        sns.heatmap(mcm[i], annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix for {label}")
        plt.savefig(f"{output_dir}/label_cms/confusion_matrix_{label}.png")
        plt.close()


def load_inference_outputs(input_file: str) -> tuple:
    data = np.load(input_file)
    logits = data["logits"]  # (n_samples, N, num_labels)
    labels = data["labels"]  # (N, L)

    probs = expit(logits)
    probs = np.clip(probs, 1e-8, 1 - 1e-8)

    mean_probs = probs.mean(axis=0)  # (N, num_labels)

    # ENTROPY-BASED UNCERTAINTY DECOMPOSITION
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

    # # VARIANCE-BASED UNCERTAINTY DECOMPOSITION
    # epistemic_unc = probs.var(axis=0)
    # aleatoric_unc = (probs * (1 - probs)).mean(axis=0)
    # total_unc = epistemic_unc + aleatoric_unc

    return mean_probs, total_unc, epistemic_unc, aleatoric_unc, labels


def plot_f1_rejection_curve(
    preds_matrix: np.ndarray,
    uncertainty: np.ndarray,
    all_labels: np.ndarray,
    label_names: list,
    output_file: str,
) -> None:
    rejection_fractions = np.linspace(0.0, 0.5, 20)

    fig, ax = plt.subplots(figsize=(12, 8))

    all_f1_scores = {label: [] for label in label_names}
    macro_f1_scores = []

    for frac in rejection_fractions:
        current_step_f1 = []

        for i, label in enumerate(label_names):
            preds = preds_matrix[:, i]
            unc_class = uncertainty[:, i]
            labels_class = all_labels[:, i]

            pos_indices = np.where(preds == 1)[0]
            neg_indices = np.where(preds == 0)[0]

            if len(pos_indices) > 0:
                sorted_pos = pos_indices[np.argsort(unc_class[pos_indices])]
                keep_pos_count = int(len(pos_indices) * (1.0 - frac))
                keep_pos = sorted_pos[:keep_pos_count]
            else:
                keep_pos = np.array([], dtype=int)

            if len(neg_indices) > 0:
                sorted_neg = neg_indices[np.argsort(unc_class[neg_indices])]
                keep_neg_count = int(len(neg_indices) * (1.0 - frac))
                keep_neg = sorted_neg[:keep_neg_count]
            else:
                keep_neg = np.array([], dtype=int)

            keep_indices = np.concatenate([keep_pos, keep_neg])

            if len(keep_indices) == 0:
                all_f1_scores[label].append(np.nan)
                continue

            filt_preds = preds[keep_indices]
            filt_labels = labels_class[keep_indices]

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
        ax.plot(
            rejection_fractions[: len(f1_series)][valid_mask],
            f1_series[valid_mask],
            alpha=0.15,
            color="gray",
        )

    macro_series = np.array(macro_f1_scores)
    valid_macro_mask = ~np.isnan(macro_series)
    ax.plot(
        rejection_fractions[: len(macro_series)][valid_macro_mask],
        macro_series[valid_macro_mask],
        color="red",
        linewidth=3,
        label="Macro F1",
    )

    ax.set_xlabel("Rejection Fraction")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1-Rejection Curve")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_uncertainty_ridge(
    uncertainty: np.ndarray, label_names: list, output_file: str, unc_type: str
) -> None:
    plt.figure(figsize=(12, 16))

    df = pd.DataFrame(uncertainty, columns=label_names)
    df_melted = df.melt(var_name="Emotion", value_name=f"{unc_type} Uncertainty")

    sns.violinplot(
        data=df_melted,
        x=f"{unc_type} Uncertainty",
        y="Emotion",
        orient="h",
        cut=0,
        density_norm="width",
        inner="quartile",
    )
    plt.title(f"Distribution of {unc_type} Uncertainty per Emotion")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_multilabel_cm(
    all_labels: np.ndarray, preds: np.ndarray, label_names: list, output_file: str
) -> None:
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
    plt.savefig(output_file)
    plt.close()


def get_report_df(
    all_labels: np.ndarray, preds: np.ndarray, label_names: list
) -> pd.DataFrame:
    report_dict = classification_report(
        all_labels, preds, target_names=label_names, zero_division=0, output_dict=True
    )
    df_report = pd.DataFrame(report_dict).transpose()
    df_report = df_report.round(3)
    df_report["support"] = df_report["support"].astype(int)
    df_report.index.name = "label"
    df_report = df_report.reset_index()
    return df_report


def main() -> None:
    config = EvalConfig()  # type:ignore
    os.makedirs(config.output_path, exist_ok=True)

    # load optimized thresholds
    with open(f"{config.model_path}/thresholds.json", "r") as f:
        thresholds = json.load(f)

    plt.rcParams["figure.figsize"] = (14, 10)
    sns.set_theme(style="whitegrid")

    # load inference outputs
    mean_probs, total_unc, epistemic_unc, aleatoric_unc, all_labels = (
        load_inference_outputs(config.input_file)
    )

    with open(f"{config.model_path}/config.json", "r") as f:
        model_config = EmCoderConfig.model_validate_json(f.read())

    label_names = [model_config.id2label[i] for i in range(model_config.num_labels)]

    # get predictions using optimized thresholds
    preds = np.zeros_like(all_labels)
    for i, label_name in enumerate(label_names):
        p_thresh = thresholds[label_name]["p"]
        preds[:, i] = (mean_probs[:, i] > p_thresh).astype(int)

    report_df = get_report_df(all_labels, preds, label_names)
    report_df.to_csv(f"{config.output_path}/classification_report.csv", index=False)
    print(report_df)

    plot_multilabel_cm(
        all_labels, preds, label_names, f"{config.output_path}/confusion_matrix.png"
    )

    plot_per_label_scatters(
        mean_probs,
        epistemic_unc,
        aleatoric_unc,
        all_labels,
        preds,
        label_names,
        config.output_path,
    )

    plot_uncertainty_ridge(
        aleatoric_unc,
        label_names,
        f"{config.output_path}/ridge_aleatoric.png",
        unc_type="Aleatoric",
    )
    plot_uncertainty_ridge(
        epistemic_unc,
        label_names,
        f"{config.output_path}/ridge_epistemic.png",
        unc_type="Epistemic",
    )

    plot_f1_rejection_curve(
        preds,
        epistemic_unc,
        all_labels,
        label_names,
        f"{config.output_path}/f1_rejection_epistemic.png",
    )
    plot_f1_rejection_curve(
        preds,
        aleatoric_unc,
        all_labels,
        label_names,
        f"{config.output_path}/f1_rejection_aleatoric.png",
    )

    plot_per_label_calibrations(all_labels, mean_probs, label_names, config.output_path)

    plot_per_label_cms(all_labels, preds, label_names, config.output_path)


if __name__ == "__main__":
    main()
