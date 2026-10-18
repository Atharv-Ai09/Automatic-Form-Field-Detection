"""
test_api.py - automatic checks for the error handling of the Flask API.

Run:  python test_api.py
Tests 1 needs a trained model; if the model is missing it is skipped
and Test 4 (prediction without a model) is checked instead.
"""
import io
import os

import cv2
import numpy as np

import config
from app import app

client = app.test_client()
results = []


def check(name, condition):
    results.append(condition)
    print(("PASS  " if condition else "FAIL  ") + name)


def png_bytes(width=400, height=300):
    image = np.full((height, width, 3), 255, np.uint8)
    cv2.putText(image, "Name:", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.rectangle(image, (150, 30), (380, 70), (0, 0, 0), 1)
    return cv2.imencode(".png", image)[1].tobytes()


def post(filename, data):
    return client.post("/predict", data={"image": (io.BytesIO(data), filename)},
                       content_type="multipart/form-data")


check("Health endpoint", client.get("/health").get_json() == {"status": "ok"})

model_exists = os.path.exists(config.MODEL_PATH)
r = post("form.png", png_bytes())
if model_exists:  # Test 1
    body = r.get_json()
    check("Test 1: valid image returns success + fields + image",
          r.status_code == 200 and body["success"] and "fields" in body and body["image"].startswith("data:image/png"))
else:             # Test 4
    check("Test 4: no model -> clear 503 message", r.status_code == 503 and "train" in r.get_json()["error"])

r = post("notes.txt", b"this is not an image")                      # Test 2
check("Test 2: invalid file rejected", r.status_code == 415 and not r.get_json()["success"])
r = post("fake.png", b"this is not an image")                       # Test 2b
check("Test 2b: corrupted .png rejected", r.status_code == 400)
r = post("picture.gif", png_bytes())                                # Test 3
check("Test 3: unsupported format rejected", r.status_code == 415)
r = client.post("/predict", data={}, content_type="multipart/form-data")
check("No file field -> 400", r.status_code == 400)

big = cv2.imencode(".png", np.full((4500, 3500, 3), 255, np.uint8))[1].tobytes()   # Test 5
r = post("big.png", big)
check("Test 5: large image handled without crashing", r.status_code in (200, 503))
r = post("huge.png", b"0" * (11 * 1024 * 1024))
check("Test 5b: file over 10 MB -> 413", r.status_code == 413)

print(f"\n{sum(results)} / {len(results)} checks passed")
