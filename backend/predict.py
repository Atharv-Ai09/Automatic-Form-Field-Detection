"""
predict.py - finds form fields in an image and classifies them.

Two-stage approach:
  Stage 1 (OpenCV)         : find candidate text/field regions (no learning involved)
  Stage 2 (neural network) : classify each region as question / answer / header / other

Command line use:  python predict.py path/to/form.png
"""
import os
import sys

import cv2
import numpy as np

import config
from model import FormFieldNet
from preprocessing import (limit_image_size, preprocess_page, region_to_features,
                           threshold_page)

# Box colours in BGR order (OpenCV uses BGR)
COLORS = {
    "question": (219, 130, 37),   # blue
    "answer": (80, 160, 40),      # green
    "header": (50, 50, 210),      # red
    "other": (140, 140, 140),     # grey
}


class ModelNotFoundError(Exception):
    """Raised when model/form_field_model.npz does not exist yet."""


_model_cache = {"mtime": None, "model": None}


def load_model():
    """Load the trained model (re-loads automatically if the file changes)."""
    if not os.path.exists(config.MODEL_PATH):
        raise ModelNotFoundError(
            "No trained model found. Prepare the dataset and run 'python train.py' first."
        )
    mtime = os.path.getmtime(config.MODEL_PATH)
    if _model_cache["mtime"] != mtime:
        _model_cache["model"] = FormFieldNet.load(config.MODEL_PATH)
        _model_cache["mtime"] = mtime
    return _model_cache["model"]


# ------------------------------------------------------------ stage 1
def detect_candidate_regions(gray):
    """
    Find text-like regions with classic image processing:
      1. threshold  -> ink becomes white
      2. dilate horizontally -> letters and words of one line join into one blob
      3. find contours -> each blob gives a bounding box
      4. drop boxes that are too small or as big as the whole page
    Returns a list of [x1, y1, x2, y2].
    """
    h, w = gray.shape[:2]
    binary = threshold_page(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 50), max(3, h // 300)))
    joined = cv2.dilate(binary, kernel)
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 15 or bh < 6:
            continue  # too small: noise
        if bh > h * 0.3 or bw * bh > 0.5 * w * h:
            continue  # too big: page border / table frame
        boxes.append([x, y, x + bw, y + bh])

    boxes.sort(key=lambda b: (b[1] // 10, b[0]))  # roughly top-to-bottom, left-to-right
    return boxes


# ------------------------------------------------------------ stage 2
def classify_regions(gray, boxes, model):
    """Run the neural network on every candidate box."""
    vectors, valid_boxes = [], []
    for box in boxes:
        vector = region_to_features(gray, box)
        if vector is not None:
            vectors.append(vector)
            valid_boxes.append(box)
    if not vectors:
        return []
    probs = model.predict_proba(np.array(vectors, dtype=np.float32))
    fields = []
    for box, p in zip(valid_boxes, probs):
        k = int(p.argmax())
        fields.append({
            "label": config.CLASS_NAMES[k],
            "confidence": round(float(p[k]), 4),  # softmax probability from the model
            "bbox": [int(v) for v in box],
        })
    return fields


# ------------------------------------------------------------ drawing
def draw_fields(image, fields):
    """Draw coloured boxes with a readable label above each one."""
    out = image.copy()
    scale = max(0.4, min(out.shape[:2]) / 1600.0 + 0.3)
    thick = max(1, int(round(scale * 2)))
    for f in fields:
        x1, y1, x2, y2 = f["bbox"]
        color = COLORS[f["label"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        text = f["label"]
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.6, 1)
        top = max(0, y1 - th - base - 2)
        cv2.rectangle(out, (x1, top), (x1 + tw + 4, top + th + base + 2), color, -1)
        cv2.putText(out, text, (x1 + 2, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    scale * 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ------------------------------------------------------------ main entry
def predict_image(image, model=None):
    """
    image: BGR image (NumPy array).
    Returns (fields, annotated_image). Coordinates refer to the returned image.
    """
    if model is None:
        model = load_model()
    image = limit_image_size(image)
    gray = preprocess_page(image)
    boxes = detect_candidate_regions(gray)
    fields = classify_regions(gray, boxes, model)
    return fields, draw_fields(image, fields)


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/form_image.png")
        sys.exit(1)
    image = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    if image is None:
        print("ERROR: could not read an image from", sys.argv[1])
        sys.exit(1)
    try:
        fields, annotated = predict_image(image)
    except ModelNotFoundError as err:
        print("ERROR:", err)
        sys.exit(1)

    print("Detected Fields:\n")
    for i, f in enumerate(fields, 1):
        print(f"Field {i}\nClass: {f['label']}\nConfidence: {f['confidence']}\nBounding Box: {f['bbox']}\n")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, "prediction_" + os.path.basename(sys.argv[1]))
    if not out_path.lower().endswith((".png", ".jpg", ".jpeg")):
        out_path += ".png"
    cv2.imwrite(out_path, annotated)
    print("Annotated image saved to", out_path)


if __name__ == "__main__":
    main()
