import "./style.css";

// Address of the Flask backend. Change it with VITE_API_URL if needed.
const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";

const COLORS = { question: "#2582db", answer: "#28a050", header: "#d23232", other: "#8c8c8c" };

const fileInput = document.getElementById("file-input");
const fileName = document.getElementById("file-name");
const preview = document.getElementById("preview");
const detectBtn = document.getElementById("detect-btn");
const message = document.getElementById("message");
const resultImage = document.getElementById("result-image");
const resultEmpty = document.getElementById("result-empty");
const fieldsList = document.getElementById("fields");
const fieldsEmpty = document.getElementById("fields-empty");
const count = document.getElementById("count");
const modelInfo = document.getElementById("model-info");
const legend = document.getElementById("legend");

// Colour legend
for (const [name, color] of Object.entries(COLORS)) {
  const item = document.createElement("span");
  item.className = "legend-item";
  const swatch = document.createElement("i");
  swatch.style.background = color;
  item.append(swatch, name);
  legend.append(item);
}

function showMessage(text, isError) {
  message.textContent = text;
  message.className = "message" + (isError ? " error" : "");
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  fileName.textContent = file.name;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  detectBtn.disabled = false;
  showMessage("", false);
});

detectBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("image", file);
  detectBtn.disabled = true;
  showMessage("Detecting fields...", false);

  try {
    const response = await fetch(`${API_URL}/predict`, { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok || !data.success) {
      throw new Error(data.error || "The server returned an error.");
    }
    showResult(data);
    showMessage(`Done. ${data.field_count} candidate regions classified.`, false);
  } catch (err) {
    const offline = err instanceof TypeError; // fetch fails when the server is not running
    showMessage(offline ? `Cannot reach the backend at ${API_URL}. Start it with: python app.py` : err.message, true);
  } finally {
    detectBtn.disabled = false;
  }
});

function showResult(data) {
  resultImage.src = data.image;
  resultImage.hidden = false;
  resultEmpty.hidden = true;

  fieldsList.replaceChildren();
  count.textContent = data.field_count;
  fieldsEmpty.hidden = data.fields.length > 0;
  data.fields.forEach((f, i) => {
    const li = document.createElement("li");
    const dot = document.createElement("i");
    dot.style.background = COLORS[f.label] || COLORS.other;
    const title = document.createElement("strong");
    title.textContent = `${i + 1}. ${f.label}`;
    const detail = document.createElement("span");
    detail.textContent = `confidence ${f.confidence.toFixed(2)}  |  box [${f.bbox.join(", ")}]`;
    li.append(dot, title, detail);
    fieldsList.append(li);
  });
}

function addInfo(label, value) {
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
  dd.textContent = value;
  modelInfo.append(dt, dd);
}

async function loadModelInfo() {
  modelInfo.replaceChildren();
  try {
    const info = await (await fetch(`${API_URL}/model-info`)).json();
    if (!info.model_trained) {
      addInfo("Status", "Backend running, model not trained yet (run python train.py)");
      return;
    }
    addInfo("Status", "Trained model loaded");
    if (info.training) {
      addInfo("Architecture", info.training.architecture);
      addInfo("Training regions", info.training.train_samples);
      addInfo("Validation accuracy", info.training.validation_accuracy);
    }
    if (info.evaluation) {
      addInfo("Test accuracy", info.evaluation.accuracy.toFixed(4));
      addInfo("Test macro F1", info.evaluation.macro.f1.toFixed(4));
    } else {
      addInfo("Test results", "Not evaluated yet (run python evaluation.py)");
    }
  } catch {
    addInfo("Status", `Backend not reachable at ${API_URL}`);
  }
}

loadModelInfo();
