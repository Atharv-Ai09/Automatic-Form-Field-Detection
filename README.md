# FormDetect — Automatic Form Field Detection and Classification using Deep Learning

**Student:** Atharv Shetty  |  **Register No.:** 8080110
**Programme:** Diploma in Artificial Intelligence and Machine Learning, GTTC Belagavi (2026)
**Guide:** K.S Shirur

FormDetect takes a scanned form image, finds text/field regions with OpenCV, classifies each region as
**question, answer, header or other** with a neural network trained on the **FUNSD** dataset, draws
coloured bounding boxes with labels, and shows the result in a web page together with the model's
confidence for every region.

The neural network is written **from scratch with NumPy** (forward propagation, backpropagation,
weighted cross-entropy loss, Adam optimizer). No TensorFlow, no PyTorch, no pre-trained weights and
no external AI API are used.

---

## Screenshots

**1. Upload a form, preview it and press *Detect Fields*** — the result image shows coloured boxes
(question = blue, answer = green, header = red, other = grey).

![Upload page and result image](screenshots/01-upload-and-result.png)

**2. Detected fields** — every region gets a label, the model's confidence and its bounding box
`[x1, y1, x2, y2]`; the field count is shown next to the heading.

![Detected fields list](screenshots/02-detected-fields.png)

**3. Model information and How it works** — loaded model, architecture and the measured validation and
test figures read from the backend.

![Model information](screenshots/03-model-information.png)

---

## How it works

```
Form Image
     |
Image Preprocessing        (grayscale, noise reduction)
     |
Candidate Region Detection (OpenCV: threshold + dilate + contours)
     |
Feature Extraction         (32x32 normalised crop + 5 position/shape numbers = 1029 values)
     |
Deep Learning Classifier   (NumPy neural network 1029 -> 128 -> 64 -> 4)
     |
Question / Answer / Header / Other  (+ softmax probability = confidence)
     |
Bounding Box Generation
     |
Flask Backend  ->  Web Interface  ->  Final Result
```

**Two-stage design**

1. **Stage 1 – candidate regions (OpenCV, no learning):** threshold the page, dilate horizontally so the
   words of a line join together, take the contours as boxes, and remove boxes that are too small or too large.
2. **Stage 2 – classification (neural network):** each box is cropped, resized to 32×32, normalised and
   combined with five geometry numbers (left, top, width, height, aspect ratio). The network outputs four
   probabilities; the largest is the predicted class and is shown as the confidence.

**Network:** `1029 -> 128 (ReLU) -> 64 (ReLU) -> 4 (Softmax)`, He initialisation, class-weighted
cross-entropy loss, Adam optimizer (learning rate 0.001), mini-batches of 64, best model chosen by the lowest
validation loss.

---

## Dataset

**FUNSD** (Form Understanding in Noisy Scanned Documents) — 199 scanned forms (149 training, 50 testing)
with annotated entities: bounding box, label and text. Labels used: `question`, `answer`, `header`, `other`.
Download: <https://guillaumejaume.github.io/FUNSD/>. The dataset is **not** included in this repository.

The 149 training forms are split **by document** into training (127) and validation (22) with a fixed seed
(42); the 50 testing forms are the test set.

| Split | Documents | Regions | question | answer | header | other |
|---|---|---|---|---|---|---|
| Training | 127 | 6,199 | 2,740 | 2,362 | 365 | 732 |
| Validation | 22 | 1,212 | 526 | 440 | 76 | 170 |
| Test | 50 | 2,332 | 1,077 | 821 | 122 | 312 |

(All numbers are printed by `prepare_dataset.py`.)

---

## Results (measured on the FUNSD test split)

Produced by `evaluation.py` from the trained model on 2,332 test regions.

| Metric | Value |
|---|---|
| Accuracy | 0.5742 (57.42%) |
| Macro precision | 0.5129 |
| Macro recall | 0.5144 |
| Macro F1-score | 0.4752 |
| Validation accuracy (best epoch) | 0.6559 |

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| question | 0.6815 | 0.6955 | 0.6884 | 1077 |
| answer | 0.8041 | 0.5201 | 0.6317 | 821 |
| header | 0.1356 | 0.5246 | 0.2155 | 122 |
| other | 0.4304 | 0.3173 | 0.3653 | 312 |

Confusion matrix (rows = true class, columns = predicted class):

| | question | answer | header | other |
|---|---|---|---|---|
| **question** | 749 | 53 | 212 | 63 |
| **answer** | 218 | 427 | 117 | 59 |
| **header** | 41 | 8 | 64 | 9 |
| **other** | 91 | 43 | 79 | 99 |

**Training behaviour:** training loss fell from 1.1469 to 0.0044 over 60 epochs, but validation loss was
lowest at epoch 5 (1.0630) and then rose to 4.5211, which is clear **overfitting**. The saved model is the
epoch 5 model. The loss graph is saved by `train.py` as `results/loss_curve.png`.

**Honest reading of the results:** the performance is modest. *Question* is recognised best, *answer* has
high precision but low recall, and *header* has low precision (many questions, answers and other regions are
wrongly labelled header). The scores use the annotated FUNSD boxes; boxes found by the OpenCV detector can
give lower performance on new forms.

---

## Project structure

```
Automatic-Form-Field-Detection/
├── backend/
│   ├── app.py                Flask API (/health, /model-info, /predict)
│   ├── config.py             shared settings (classes, sizes, paths, seed)
│   ├── prepare_dataset.py    reads FUNSD annotations, builds train/validation/test data
│   ├── preprocessing.py      image preprocessing and feature extraction
│   ├── model.py              NumPy neural network (forward, backprop, Adam)
│   ├── train.py              training, best-model saving, loss curve
│   ├── evaluation.py         accuracy, precision, recall, F1, confusion matrix
│   ├── predict.py            candidate detection + classification + drawing
│   ├── test_api.py           automatic API error-handling checks
│   ├── requirements.txt
│   ├── dataset/{raw,train,validation,test}/
│   ├── model/                form_field_model.npz, model_info.json (created by train.py)
│   └── results/              loss_curve.png, confusion_matrix.png, evaluation_report.txt, metrics.json
├── frontend/                 Vite + HTML/CSS/JavaScript web interface
├── screenshots/              images used in this README
├── README.md
├── REPORT.md
└── .gitignore
```

---

## Setup and run

**Requirements:** Python 3.9+ and Node.js (for the frontend). No GPU needed.

**1. Download FUNSD** and extract it so that these folders exist:
```
backend/dataset/raw/dataset/training_data/{images,annotations}
backend/dataset/raw/dataset/testing_data/{images,annotations}
```

**2. Install the Python packages**
```
cd backend
python -m venv venv
venv\Scripts\activate            # Windows   (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

**3. Prepare the dataset** (prints the real class counts)
```
python prepare_dataset.py
```

**4. Train** (optional: `--epochs 100 --lr 0.001`)
```
python train.py
```

**5. Evaluate** on the test split
```
python evaluation.py
```

**6. Start the Flask backend** (http://127.0.0.1:5000)
```
python app.py
```

**7. Start the frontend** (new terminal)
```
cd frontend
npm install
npm run dev
```
Open the address Vite prints (normally http://localhost:5173). If the backend runs elsewhere, set
`VITE_API_URL` (for example in `frontend/.env`).

**Command-line prediction:** `python predict.py path\to\form.png` prints the fields and saves an annotated
image in `results/`.

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | `{"status": "ok"}` |
| `/model-info` | GET | whether a model exists, plus training and evaluation figures if available |
| `/predict` | POST | multipart field `image`; returns detected fields and the annotated image |

Example `/predict` response:
```json
{
  "success": true,
  "field_count": 8,
  "image_size": [762, 1000],
  "fields": [
    { "label": "question", "confidence": 0.62, "bbox": [97, 86, 359, 100] }
  ],
  "image": "data:image/png;base64,..."
}
```
`confidence` is the softmax probability produced by the model. Errors return
`{"success": false, "error": "..."}` with status 400 (invalid image), 413 (file over 10 MB),
415 (unsupported type; allowed: PNG, JPG, JPEG, BMP) or 503 (model not trained yet).

---

## Testing

Run the automatic API checks with `python test_api.py`, or test by hand in the web page:

| Test | Action | Expected result |
|---|---|---|
| 1 | Upload a valid form image | Result image, field list and count appear |
| 2 | Upload a `.txt` file or a corrupted `.png` | Message: unsupported file type / not a valid image |
| 3 | Upload a `.gif` | Message: unsupported file type |
| 4 | Remove the contents of `backend/model/`, then detect | Message: no trained model, run `train.py` |
| 5 | Upload a very large image (e.g. 4500×3500) | Image is shrunk to 2000 px and processed; files over 10 MB are rejected |

---

## Viva quick answers

- **Deep learning:** machine learning with multi-layer neural networks that learn from examples.
- **Form field detection:** finding where the labels, answers and headings are on a form.
- **Bounding box:** a rectangle `[x1, y1, x2, y2]` around a region.
- **Classification:** choosing one of four classes for each region.
- **Preprocessing:** cleaning and standardising the image before the model sees it.
- **Training / validation / testing:** learning from data / checking progress and choosing the best model / final score on unseen data.
- **Accuracy:** correct predictions ÷ all predictions.
- **Precision:** of regions predicted as a class, how many really were that class.
- **Recall:** of regions that really are a class, how many were found.
- **F1-score:** the balance (harmonic mean) of precision and recall.

---

## Limitations and future work

- FUNSD is small (199 forms) and the classes are unequal; the model overfits after a few epochs.
- The OpenCV detector is simple and may merge or split regions; there is no OCR, so the model never reads the words.
- Training uses annotated boxes but prediction uses detected boxes, so real-world results can be lower than the test score.
- Ideas: early stopping, weight decay or dropout, data augmentation, OCR words as extra features, a convolutional network, linking each question to its answer.

---

## Credits

- Dataset: G. Jaume, H. K. Ekenel, J.-P. Thiran, *FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents*, ICDAR-OST 2019.
- Libraries: NumPy, OpenCV, Matplotlib, Flask, Vite.
