"""
app.py - Flask backend.

Endpoints:
  GET  /health      -> {"status": "ok"}
  GET  /model-info  -> information about the trained model (if it exists)
  POST /predict     -> multipart/form-data with field "image"

Run:  python app.py      (server on http://127.0.0.1:5000)
"""
import base64
import json
import os

import cv2
from flask import Flask, jsonify, request

import config
from predict import ModelNotFoundError, load_model, predict_image
from preprocessing import load_image_from_bytes

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
MAX_UPLOAD_MB = 10

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


def error(message, status):
    return jsonify({"success": False, "error": message}), status


@app.after_request
def allow_frontend(response):
    """Let the frontend (a different port on localhost) call this API."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.errorhandler(413)
def too_large(_):
    return error(f"File is too large. Maximum upload size is {MAX_UPLOAD_MB} MB.", 413)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/model-info")
def model_info():
    info = {"model_trained": os.path.exists(config.MODEL_PATH), "training": None, "evaluation": None}
    if os.path.exists(config.MODEL_INFO_PATH):
        with open(config.MODEL_INFO_PATH) as f:
            info["training"] = json.load(f)
    metrics_path = os.path.join(config.RESULTS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            info["evaluation"] = json.load(f)
    return jsonify(info)


@app.route("/predict", methods=["POST"])
def predict():
    # 1. Receive and validate the upload
    if "image" not in request.files:
        return error("No file received. Choose an image and try again.", 400)
    upload = request.files["image"]
    if upload.filename == "":
        return error("No file selected.", 400)
    extension = os.path.splitext(upload.filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        return error("Unsupported file type. Please upload a PNG, JPG, JPEG or BMP image.", 415)

    # 2. Read the image
    image = load_image_from_bytes(upload.read())
    if image is None:
        return error("The file is not a valid image or is corrupted.", 400)

    # 3-7. Preprocess, detect, classify, draw
    try:
        model = load_model()
    except ModelNotFoundError as err:
        return error(str(err), 503)
    fields, annotated = predict_image(image, model)

    # 8-9. Return JSON with the predictions and the processed image
    ok, buffer = cv2.imencode(".png", annotated)
    if not ok:
        return error("Could not create the result image.", 500)
    data_url = "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")
    return jsonify({
        "success": True,
        "field_count": len(fields),
        "image_size": [int(annotated.shape[1]), int(annotated.shape[0])],
        "fields": fields,
        "image": data_url,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
