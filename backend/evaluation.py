"""
evaluation.py - measures the trained model on the TEST split.

All numbers are calculated from the model's predictions; nothing is typed in by hand.
Outputs (in results/):
    confusion_matrix.png, evaluation_report.txt, metrics.json

Run:  python evaluation.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from model import FormFieldNet


def confusion_matrix(y_true, y_pred, n_classes):
    """cm[i][j] = number of samples of true class i predicted as class j."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def safe_divide(a, b):
    return float(a) / b if b > 0 else 0.0


def main():
    if not os.path.exists(config.MODEL_PATH):
        raise SystemExit("No trained model found. Run 'python train.py' first.")
    test_file = os.path.join(config.TEST_DIR, "data.npz")
    if not os.path.exists(test_file):
        raise SystemExit("No test data found. Run 'python prepare_dataset.py' first.")

    data = np.load(test_file)
    X, y = data["X"], data["y"]
    if len(y) == 0:
        raise SystemExit("Test set is empty.")

    net = FormFieldNet.load(config.MODEL_PATH)
    pred = net.predict(X)
    n = len(config.CLASS_NAMES)
    cm = confusion_matrix(y, pred, n)

    accuracy = safe_divide(np.trace(cm), cm.sum())
    per_class = {}
    for i, name in enumerate(config.CLASS_NAMES):
        tp = cm[i, i]
        precision = safe_divide(tp, cm[:, i].sum())  # of all predicted as i, how many were right
        recall = safe_divide(tp, cm[i, :].sum())     # of all real i, how many were found
        f1 = safe_divide(2 * precision * recall, precision + recall)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(cm[i, :].sum())}

    # Macro average: simple mean over the classes that really occur in the test set
    present = [c for c in config.CLASS_NAMES if per_class[c]["support"] > 0]
    macro = {k: float(np.mean([per_class[c][k] for c in present])) for k in ("precision", "recall", "f1")}

    # ---- text report
    lines = [
        "EVALUATION REPORT (FUNSD test split, annotated regions)",
        "=" * 56,
        f"Test samples : {len(y)}",
        f"Accuracy     : {accuracy:.4f}",
        "",
        f"{'class':<10}{'precision':>11}{'recall':>9}{'f1-score':>10}{'support':>9}",
    ]
    for name in config.CLASS_NAMES:
        m = per_class[name]
        lines.append(f"{name:<10}{m['precision']:>11.4f}{m['recall']:>9.4f}{m['f1']:>10.4f}{m['support']:>9d}")
    lines.append(f"{'macro avg':<10}{macro['precision']:>11.4f}{macro['recall']:>9.4f}{macro['f1']:>10.4f}{len(y):>9d}")
    lines += ["", "Confusion matrix (rows = true class, columns = predicted class):",
              " " * 10 + "".join(f"{c:>10}" for c in config.CLASS_NAMES)]
    for i, name in enumerate(config.CLASS_NAMES):
        lines.append(f"{name:<10}" + "".join(f"{v:>10d}" for v in cm[i]))
    lines += ["", "Note: these results use the ground-truth boxes of FUNSD.",
              "Boxes found automatically by the OpenCV detector in predict.py are not scored here."]

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    report = "\n".join(lines)
    with open(os.path.join(config.RESULTS_DIR, "evaluation_report.txt"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(config.RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump({"test_samples": int(len(y)), "accuracy": accuracy, "macro": macro, "per_class": per_class}, f, indent=2)

    # ---- confusion matrix picture
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(config.CLASS_NAMES)
    ax.set_yticks(range(n)); ax.set_yticklabels(config.CLASS_NAMES)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title("Confusion matrix (test set)")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "confusion_matrix.png"), dpi=120)
    plt.close()

    print(report)
    print("\nSaved: results/evaluation_report.txt, confusion_matrix.png, metrics.json")


if __name__ == "__main__":
    main()
