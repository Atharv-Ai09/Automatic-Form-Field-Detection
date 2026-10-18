"""
preprocessing.py - image preprocessing shared by training AND prediction.

Using the same functions in both places keeps the data consistent,
which is important for a neural network.
"""
import cv2
import numpy as np

from config import REGION_SIZE, MAX_IMAGE_SIDE


# ---------------------------------------------------------------- loading
def load_image(path):
    """Read an image file from disk (BGR colour). Returns None if it fails."""
    return cv2.imread(path, cv2.IMREAD_COLOR)


def load_image_from_bytes(data):
    """Decode image bytes (e.g. an uploaded file). Returns None if not an image."""
    array = np.frombuffer(data, dtype=np.uint8)
    if array.size == 0:
        return None
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------- resizing
def limit_image_size(image, max_side=MAX_IMAGE_SIDE):
    """Shrink very large images (keeping the aspect ratio)."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------- cleaning the page
def to_grayscale(image):
    """Convert a colour image to grayscale."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def reduce_noise(gray):
    """Light Gaussian blur removes small scanner noise."""
    return cv2.GaussianBlur(gray, (3, 3), 0)


def preprocess_page(image):
    """
    Full-page preprocessing: colour -> grayscale -> noise reduction.
    The result is the grayscale page used for BOTH training and prediction.
    """
    return reduce_noise(to_grayscale(image))


def threshold_page(gray):
    """
    Adaptive threshold: ink becomes white (255) and paper black (0).
    Used by the candidate region detector in predict.py.
    """
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15
    )


# --------------------------------------------------- region -> features
def crop_region(gray, box, pad=2):
    """Crop a bounding box [x1, y1, x2, y2] out of the grayscale page."""
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return gray[y1:y2, x1:x2]


def normalize_region(crop):
    """
    Resize the crop to REGION_SIZE x REGION_SIZE and scale pixels to 0..1.
    Pixels are inverted so that ink is close to 1 and white paper is close to 0.
    """
    small = cv2.resize(crop, (REGION_SIZE, REGION_SIZE), interpolation=cv2.INTER_AREA)
    return 1.0 - small.astype(np.float32) / 255.0


def geometry_features(box, page_w, page_h):
    """
    Five simple numbers describing where the region is and what shape it has:
    left, top, width, height (all as fractions of the page) and aspect ratio.
    A question usually sits at the left, an answer to its right, a header near the top.
    """
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    aspect = min(bw / float(bh), 20.0) / 20.0
    return np.array(
        [x1 / page_w, y1 / page_h, bw / page_w, bh / page_h, aspect], dtype=np.float32
    )


def region_to_features(gray, box):
    """
    Turn one region into the vector given to the neural network:
    1024 normalised pixels + 5 geometry numbers  ->  1029 values.
    Returns None if the box is invalid.
    """
    page_h, page_w = gray.shape[:2]
    crop = crop_region(gray, box)
    if crop is None:
        return None
    pixels = normalize_region(crop).flatten()
    geometry = geometry_features(box, page_w, page_h)
    return np.concatenate([pixels, geometry]).astype(np.float32)
