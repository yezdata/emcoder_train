import json

import numpy as np
from scipy.special import expit
from sklearn.metrics import f1_score

from src.config_models import ClassifierConfig


def optimize_thresholds(
    mean_probs: np.ndarray,
    all_labels: np.ndarray,
    label_names: list,
) -> dict:
    """Per-class threshold optimization"""

    p_space = np.linspace(0.1, 0.9, 15)

    best_thresholds = {}

    for i, label_name in enumerate(label_names):
        best_f1 = 0.0

        for p in p_space:
            preds = (mean_probs[:, i] > p).astype(int)
            f1 = f1_score(all_labels[:, i], preds, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[label_name] = {
                    "p": float(p),
                    "f1": float(f1),
                }

        print(
            f"Optimized {label_name}: F1={best_f1:.4f} (p={best_thresholds[label_name]['p']})"
        )

    return best_thresholds


def main(input_file: str, model_path: str) -> None:
    data = np.load(input_file)

    logits = data["logits"]  # (n_samples, N, num_labels)
    all_labels = data["labels"]  # (N, num_labels)

    probs = expit(logits)
    probs = np.clip(probs, 1e-8, 1 - 1e-8)

    mean_probs = probs.mean(axis=0)  # (N, num_labels)

    with open(f"{model_path}/config.json", "r") as f:
        model_config = ClassifierConfig.model_validate_json(f.read())

    label_names = [model_config.id2label[i] for i in range(model_config.num_labels)]

    best_thresholds = optimize_thresholds(mean_probs, all_labels, label_names)

    with open(f"{model_path}/thresholds.json", "w") as f:
        json.dump(best_thresholds, f, indent=4)


if __name__ == "__main__":
    INPUT_FILE = "outputs/v1_5/eval_outputs.npz"
    EMCODER_PATH = "models/v1_5/emcoder/epoch_4"

    main(INPUT_FILE, EMCODER_PATH)
