# Automatic Form Field Detection and Classification using Deep Learning

**Student:** ATHARV SHETTY  **Register No.:** 8080110  (IA-3 mini project)

FormDetect takes a scanned form image, finds text/field regions, classifies each region as
**question, answer, header or other** with a neural network trained on the **FUNSD** dataset,
draws coloured bounding boxes with labels, and shows the result in a web page.

> **Honest status:** this repository contains code only. No trained model, no dataset and
> no evaluation results are included. All accuracy numbers come from running
> `train.py` and `evaluation.py` yourself on the real FUNSD data.

## System architecture

```
Form Image
     |
Image Preprocessing        (grayscale, noise reduction)
     |
Candidate Region Detection (OpenCV: threshold + dilate + contours)
     |
Feature Extraction         (32x32 normalised crop + 5 position/shape numbers)
     |
Deep Learning Classifier   (NumPy neural network, trained on FUNSD)
     |
Question / Answer / Header / Other
     |
Bounding Box Generation
     |
Flask Backend
     |
Web Interface
     |
Final Result
```

**Two-stage design.** A fully custom object detector is too complex for this scope, so:
- Stage 1 finds candidate regions with classic OpenCV (no learning).
- Stage 2 is a neural network (written from scratch with NumPy, no pretrained weights) that classifies each region.

The network is trained on the **annotated FUNSD boxes**, but at prediction time it receives
**boxes found by OpenCV**. These differ slightly (e.g. OpenCV may split one FUNSD entity into
several boxes), so real-world results on uploaded forms can be lower than the test-set score.

## Project structure

```
backend/   app.py  config.py  prepare_dataset.py  preprocessing.py  model.py
           train.py  evaluation.py  predict.py  test_api.py  requirements.txt
           dataset/{raw,train,validation,test}   model/   results/
frontend/  package.json  index.html  src/(main.js, style.css)  public/
README.md  REPORT.md  .gitignore
```

## Setup and run

**Step 1 - Download FUNSD.** From the official page <https://guillaumejaume.github.io/FUNSD/>
download `dataset.zip` (199 scanned forms: 149 training, 50 testing).

**Step 2 - Place it.** Extract it so you have
`backend/dataset/raw/dataset/training_data/{images,annotations}` and
`backend/dataset/raw/dataset/testing_data/{images,annotations}`.

**Step 3 - Install Python packages** (Python 3.9+):
```
cd backend
python -m venv venv
venv\Scripts\activate          # Windows      (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

**Step 4 - Prepare the dataset** (prints the real class counts):
```
python prepare_dataset.py
```
**Step 5 - Train** (optional: `--epochs 100 --lr 0.001`):
```
python train.py
```
**Step 6 - Evaluate** on the test split:
```
python evaluation.py
```
**Step 7 - Start Flask** (http://127.0.0.1:5000):
```
python app.py
```
**Step 8 - Start the frontend** (new terminal; needs Node.js):
```
cd frontend
npm install
npm run dev
```
Open the address Vite prints (normally http://localhost:5173).
If the backend runs elsewhere, set `VITE_API_URL` (e.g. in `frontend/.env`).

Command-line prediction: `python predict.py path/to/form.png` (saves an annotated image in `results/`).

## Outputs created by you (not shipped)
- `model/form_field_model.npz`, `model/model_info.json`
- `results/loss_curve.png`, `results/confusion_matrix.png`, `results/evaluation_report.txt`, `results/metrics.json`
- `dataset/<split>/data.npz` and `annotations.csv`

## API
- `GET /health` -> `{"status": "ok"}`
- `GET /model-info` -> training/evaluation info if available
- `POST /predict` (multipart field `image`) ->
  `{"success": true, "field_count": N, "fields": [{"label", "confidence", "bbox": [x1,y1,x2,y2]}], "image": "data:image/png;base64,..."}`
  Errors return `{"success": false, "error": "..."}` with status 400, 413, 415 or 503.
  `confidence` is the softmax probability produced by the model.

## Testing procedures
Run `python test_api.py` for the automatic checks, or test by hand in the web page:

| Test | Action | Expected result |
|---|---|---|
| 1 | Upload a valid form image (after training) | Result image, field list and count appear |
| 2 | Upload a `.txt` file, or a corrupted `.png` | Message: unsupported file type / not a valid image |
| 3 | Upload a `.gif` | Message: unsupported file type (PNG, JPG, JPEG, BMP only) |
| 4 | Delete `backend/model/` contents, then detect | Message: no trained model found, run `python train.py` |
| 5 | Upload a very large image (e.g. 4500x3500) | It is shrunk to 2000 px and processed; files over 10 MB give "file is too large" |

## How the Project Works
1. **Input** - the user uploads a form image in the browser.
2. **Preprocessing** - converted to grayscale, noise reduced (`preprocessing.py`). The same code is used in training and prediction.
3. **Candidate field detection** - the page is thresholded (ink = white), dilated so words of one line merge, and contours give bounding boxes (`predict.py`).
4. **Feature extraction** - each box is cropped, resized to 32x32, pixels scaled to 0-1, plus 5 numbers: left, top, width, height, aspect ratio.
5. **Deep learning classification** - a neural network (1029 -> 128 -> 64 -> 4) gives four probabilities (`model.py`).
6. **Prediction** - the class with the highest probability is the label; that probability is the confidence.
7. **Bounding box drawing** - a coloured rectangle and label are drawn per region (blue question, green answer, red header, grey other).
8. **Result display** - Flask sends the drawn image and JSON to the web page.

## Viva questions - simple answers
- **Deep learning:** machine learning with neural networks that have several layers and learn patterns from examples.
- **Form field detection:** finding where the fields (labels, boxes, headings) are on a form.
- **Bounding box:** a rectangle `[x1, y1, x2, y2]` around an object.
- **Classification:** choosing one category for an input (here one of four).
- **Preprocessing:** cleaning and standardising data before the model sees it.
- **Training:** the network sees labelled examples and adjusts its weights (backpropagation + Adam) to reduce the loss.
- **Validation:** data not used for weight updates, used during training to check progress and pick the best model.
- **Testing:** completely unseen data used once at the end for the final score.
- **Accuracy:** correct predictions / all predictions.
- **Precision:** of regions predicted as a class, how many really were that class.
- **Recall:** of regions that really are a class, how many were found.
- **F1-score:** the balance (harmonic mean) of precision and recall.

## Known limitations
- FUNSD is small (199 forms), and the "other" and "header" classes have fewer examples than question/answer.
- OpenCV candidate detection is simple and can merge or split regions; it does not read text (no OCR).
- The classifier uses image shape and position only, not the meaning of words.
- Evaluation scores use FUNSD's ground-truth boxes, not OpenCV-detected boxes.

## Verification performed during development
`python -m py_compile` passed for every Python file. The full pipeline (prepare, train, evaluate,
predict, Flask API tests) was run only on a small **synthetic** stand-in dataset to check the code
works; those numbers are meaningless and are not included. It has **not** been run on real FUNSD
data, and `npm install` / `npm run build` could not be run in the development sandbox (no npm
registry access), so please run them on your machine.
