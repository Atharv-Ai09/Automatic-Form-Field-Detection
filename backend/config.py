"""
config.py - shared settings used by every script.
Keeping them in one place guarantees that training and prediction
use exactly the same class names, region size and paths.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder where you place the downloaded FUNSD dataset (see README)
RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw")

# Folders where prepare_dataset.py writes the prepared data
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "validation")
TEST_DIR = os.path.join(DATASET_DIR, "test")

MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "form_field_model.npz")
MODEL_INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")

RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Class names in the order of their numeric IDs
CLASS_NAMES = ["question", "answer", "header", "other"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# Every cropped region is resized to REGION_SIZE x REGION_SIZE pixels
REGION_SIZE = 32

# Number of input values for the neural network:
# REGION_SIZE*REGION_SIZE pixels + 5 position/shape numbers
NUM_GEOMETRY_FEATURES = 5
INPUT_SIZE = REGION_SIZE * REGION_SIZE + NUM_GEOMETRY_FEATURES

# Fixed random seed so results can be repeated
SEED = 42

# Fraction of the FUNSD *training documents* kept aside for validation
VAL_FRACTION = 0.15

# Very large uploaded images are shrunk so that the longest side is at most this
MAX_IMAGE_SIDE = 2000
