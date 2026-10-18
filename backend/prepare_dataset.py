"""
prepare_dataset.py - turns the FUNSD dataset into training data.

What it does:
 1. Finds the FUNSD folders (training_data / testing_data) inside dataset/raw
 2. Reads every annotation JSON and matches it with its image
 3. Extracts bounding box, label (question/answer/header/other) and text
 4. Converts labels to class IDs (0..3)
 5. Crops each region, resizes it, normalises pixels (see preprocessing.py)
 6. Splits the data: FUNSD training documents -> train + validation,
    FUNSD testing documents -> test.  The split is done per DOCUMENT
    so that regions of one form never appear in two different splits.
 7. Prints the real class counts

Run:  python prepare_dataset.py
"""
import csv
import json
import os
import sys

import numpy as np

import config
from preprocessing import load_image, preprocess_page, region_to_features


def find_folder(root, name):
    """Search for a folder called `name` anywhere inside root."""
    for folder, subfolders, _ in os.walk(root):
        if name in subfolders:
            return os.path.join(folder, name)
    return None


def find_image(images_dir, stem):
    for ext in (".png", ".jpg", ".jpeg"):
        path = os.path.join(images_dir, stem + ext)
        if os.path.exists(path):
            return path
    return None


def collect_documents(split_dir):
    """Return a list of (image_path, annotation_path) pairs for one FUNSD folder."""
    ann_dir = os.path.join(split_dir, "annotations")
    img_dir = os.path.join(split_dir, "images")
    pairs = []
    for name in sorted(os.listdir(ann_dir)):
        if not name.endswith(".json"):
            continue
        image_path = find_image(img_dir, os.path.splitext(name)[0])
        if image_path is None:
            print("  warning: no image for", name)
            continue
        pairs.append((image_path, os.path.join(ann_dir, name)))
    return pairs


def process_documents(pairs):
    """Read annotations and build feature vectors for all regions of the documents."""
    features, labels, rows = [], [], []
    for image_path, ann_path in pairs:
        image = load_image(image_path)
        if image is None:
            print("  warning: cannot read image", image_path)
            continue
        gray = preprocess_page(image)

        with open(ann_path, "r", encoding="utf-8") as f:
            annotation = json.load(f)

        for item in annotation.get("form", []):
            label = item.get("label")
            if label not in config.CLASS_TO_ID:
                continue  # ignore unknown labels, never guess
            box = item.get("box")
            if not box or len(box) != 4:
                continue
            vector = region_to_features(gray, box)
            if vector is None:
                continue  # empty / invalid box
            features.append(vector)
            labels.append(config.CLASS_TO_ID[label])
            rows.append([os.path.basename(image_path), *box, label, item.get("text", "")])
    X = np.array(features, dtype=np.float32) if features else np.zeros((0, config.INPUT_SIZE), np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y, rows


def save_split(folder, name, X, y, rows):
    os.makedirs(folder, exist_ok=True)
    np.savez_compressed(os.path.join(folder, "data.npz"), X=X, y=y)
    with open(os.path.join(folder, "annotations.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "x1", "y1", "x2", "y2", "label", "text"])
        writer.writerows(rows)
    print(f"\n{name}: {len(y)} regions")
    for i, class_name in enumerate(config.CLASS_NAMES):
        print(f"  {class_name}: {int((y == i).sum())}")


def main():
    train_dir = find_folder(config.RAW_DIR, "training_data")
    test_dir = find_folder(config.RAW_DIR, "testing_data")
    if train_dir is None or test_dir is None:
        print("ERROR: FUNSD dataset not found.")
        print("Download it and extract it inside:", config.RAW_DIR)
        print("It must contain 'training_data' and 'testing_data' folders,")
        print("each with 'images' and 'annotations' sub-folders. See README.md.")
        sys.exit(1)

    print("Using FUNSD folders:\n ", train_dir, "\n ", test_dir)
    train_pairs = collect_documents(train_dir)
    test_pairs = collect_documents(test_dir)
    print(f"Documents found: {len(train_pairs)} training, {len(test_pairs)} testing")

    # Split the training DOCUMENTS into train / validation using a fixed seed
    rng = np.random.RandomState(config.SEED)
    order = rng.permutation(len(train_pairs))
    n_val = max(1, int(round(len(train_pairs) * config.VAL_FRACTION)))
    val_pairs = [train_pairs[i] for i in order[:n_val]]
    tr_pairs = [train_pairs[i] for i in order[n_val:]]
    print(f"Split by document: {len(tr_pairs)} train, {len(val_pairs)} validation, {len(test_pairs)} test")

    for name, folder, pairs in [
        ("TRAIN", config.TRAIN_DIR, tr_pairs),
        ("VALIDATION", config.VAL_DIR, val_pairs),
        ("TEST", config.TEST_DIR, test_pairs),
    ]:
        X, y, rows = process_documents(pairs)
        save_split(folder, name, X, y, rows)

    print("\nDone. Next step: python train.py")


if __name__ == "__main__":
    main()
