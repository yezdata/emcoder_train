import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.config_models import ClassifierConfig


def optimize_thresholds(
    mean_probs: np.ndarray,
    epistemic_unc: np.ndarray,
    all_labels: np.ndarray,
    label_names: list,
) -> dict:
    """Per-class threshold optimization"""
    p_space = np.linspace(0.1, 0.9, 15)
    u_space = np.linspace(0.05, 0.25, 19)

    best_thresholds = {}

    for i, label_name in enumerate(label_names):
        best_f1 = 0.0

        for p in p_space:
            for u in u_space:
                preds = ((mean_probs[:, i] > p) & (epistemic_unc[:, i] < u)).astype(int)
                f1 = f1_score(all_labels[:, i], preds, zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_thresholds[label_name] = {
                        "p": float(p),
                        "u": float(u),
                        "f1": float(f1),
                    }

        print(
            f"Optimized {label_name}: F1={best_f1:.4f} (p={best_thresholds[label_name]['p']}, u={best_thresholds[label_name]['u']})"
        )

    return best_thresholds


def main(input_file: str, model_path: str) -> None:
    df = pd.read_parquet(input_file)

    mean_probs = np.stack(df["mean_probs"].values)
    epistemic_unc = np.stack(df["epistemic_unc"].values)
    all_labels = np.stack(df["labels"].values)

    with open(f"{model_path}/config.json", "r") as f:
        model_config = ClassifierConfig.model_validate_json(f.read())

    label_names = [model_config.id2label[i] for i in range(model_config.num_labels)]

    best_thresholds = optimize_thresholds(
        mean_probs, epistemic_unc, all_labels, label_names
    )

    with open(f"{model_path}/thresholds.json", "w") as f:
        json.dump(best_thresholds, f, indent=4)


if __name__ == "__main__":
    INPUT_FILE = "outputs/v1/val_outputs.parquet"
    EMCODER_PATH = "models/v1/emcoder/epoch_10"

    main(INPUT_FILE, EMCODER_PATH)
