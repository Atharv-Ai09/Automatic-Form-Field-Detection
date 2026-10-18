"""
train.py - trains the neural network on the prepared FUNSD regions.

Run:  python train.py            (default 60 epochs)
      python train.py --epochs 100 --lr 0.001
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # draw graphs without opening a window
import matplotlib.pyplot as plt
import numpy as np

import config
from model import FormFieldNet


def load_split(folder):
    path = os.path.join(folder, "data.npz")
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}. Run 'python prepare_dataset.py' first.")
    data = np.load(path)
    return data["X"], data["y"]


def accuracy(net, X, y):
    return float((net.predict(X) == y).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    X_train, y_train = load_split(config.TRAIN_DIR)
    X_val, y_val = load_split(config.VAL_DIR)
    if len(y_train) == 0 or len(y_val) == 0:
        raise SystemExit("Prepared data is empty. Check the dataset and re-run prepare_dataset.py.")
    print(f"Training samples: {len(y_train)}   Validation samples: {len(y_val)}")

    # Rare classes get a bigger weight so the network does not ignore them
    counts = np.bincount(y_train, minlength=len(config.CLASS_NAMES)).astype(np.float64)
    class_weights = (len(y_train) / (len(counts) * np.maximum(counts, 1))).astype(np.float32)
    print("Class weights:", dict(zip(config.CLASS_NAMES, np.round(class_weights, 3))))

    net = FormFieldNet([config.INPUT_SIZE, 128, 64, len(config.CLASS_NAMES)], seed=config.SEED)
    rng = np.random.RandomState(config.SEED)

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    train_losses, val_losses = [], []
    best_val, best_epoch = float("inf"), 0

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(y_train))  # shuffle every epoch
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            net.train_step(X_train[idx], y_train[idx], class_weights, args.lr)

        train_loss = net.compute_loss(X_train, y_train, class_weights)
        val_loss = net.compute_loss(X_val, y_val, class_weights)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        marker = ""
        if val_loss < best_val:  # keep the best model (lowest validation loss)
            best_val, best_epoch = val_loss, epoch
            net.save(config.MODEL_PATH, {"epoch": epoch, "class_names": config.CLASS_NAMES})
            marker = "  <- best, model saved"
        print(f"Epoch {epoch:3d}/{args.epochs}  train loss {train_loss:.4f}  val loss {val_loss:.4f}{marker}")

    # Reload the best model to report its validation accuracy
    best = FormFieldNet.load(config.MODEL_PATH)
    info = {
        "architecture": "MLP 1029 -> 128 -> 64 -> 4 (NumPy, trained from scratch)",
        "classes": config.CLASS_NAMES,
        "epochs_run": args.epochs,
        "best_epoch": best_epoch,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "best_validation_loss": round(best_val, 4),
        "validation_accuracy": round(accuracy(best, X_val, y_val), 4),
    }
    with open(config.MODEL_INFO_PATH, "w") as f:
        json.dump(info, f, indent=2)

    # Loss graph (real values recorded above)
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, args.epochs + 1), train_losses, label="Training loss")
    plt.plot(range(1, args.epochs + 1), val_losses, label="Validation loss")
    plt.axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch ({best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (weighted cross-entropy)")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "loss_curve.png"), dpi=120)
    plt.close()

    print("\nBest model saved to", config.MODEL_PATH)
    print("Loss graph saved to results/loss_curve.png")
    print("Next step: python evaluation.py")


if __name__ == "__main__":
    main()
