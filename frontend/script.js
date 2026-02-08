// frontend/script.js
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');

const origPreview = document.getElementById('origPreview');
const prePreview = document.getElementById('prePreview');
const origPlaceholder = document.getElementById('origPlaceholder');
const prePlaceholder = document.getElementById('prePlaceholder');
const predPlaceholder = document.getElementById('predPlaceholder');

const predictionCard = document.getElementById('predictionCard');
const predLabel = document.getElementById('predLabel');
const predConf = document.getElementById('predConf');
const tryAgainBtn = document.getElementById('tryAgain');

const statusPanel = document.getElementById('statusPanel');
const loadingSpinner = document.getElementById('loadingSpinner');
const statusText = document.getElementById('statusText');

const API_BASE = "http://127.0.0.1:8000";
const API_PREDICT = API_BASE + "/predict";

// Reset UI to initial state
function resetUI() {
  // original
  origPreview.src = "";
  origPreview.classList.add('hidden');
  origPlaceholder.innerText = "No image";

  // preprocessed
  prePreview.src = "";
  prePreview.classList.add('hidden');
  prePlaceholder.innerText = "Waiting";

  // prediction
  predictionCard.classList.add('hidden');
  predPlaceholder.innerHTML = 'No result<br><span style="font-size:12px;color:#6ea9ae">Confidence: 0%</span>';

  // status
  statusPanel.classList.add('hidden');
  loadingSpinner.classList.remove('hidden');
  statusText.textContent = "Analyzing...";
}
resetUI();

// interactions
browseBtn.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });
uploadArea.addEventListener('click', () => fileInput.click());

;['dragenter','dragover'].forEach(ev => {
  uploadArea.addEventListener(ev, (e)=>{ e.preventDefault(); uploadArea.classList.add('dragover'); });
});
;['dragleave','drop','dragend'].forEach(ev => {
  uploadArea.addEventListener(ev, (e)=>{ e.preventDefault(); uploadArea.classList.remove('dragover'); });
});

uploadArea.addEventListener('drop', (e) => {
  const dt = e.dataTransfer;
  if (!dt) return;
  const files = dt.files;
  if (files.length) handleFile(files[0]);
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) handleFile(e.target.files[0]);
});

tryAgainBtn && tryAgainBtn.addEventListener('click', () => resetUI());

async function handleFile(file) {
  if (!file.type.startsWith('image/')) { alert('Please upload an image'); return; }

  // show original instantly
  const objectUrl = URL.createObjectURL(file);
  origPreview.src = objectUrl;
  origPreview.classList.remove('hidden');
  origPlaceholder.innerText = "";

  // show status
  statusPanel.classList.remove('hidden');
  loadingSpinner.classList.remove('hidden');
  statusText.textContent = "Analyzing...";

  // send to backend
  const form = new FormData();
  form.append('file', file);

  try {
    const resp = await fetch(API_PREDICT, { method: 'POST', body: form });
    if (!resp.ok) {
      const err = await resp.json().catch(()=>({detail:resp.statusText}));
      throw new Error(err.detail || 'Server error');
    }
    const data = await resp.json();

    // set preprocessed image
    if (data.preprocessed_image) {
      prePreview.src = data.preprocessed_image;
      prePreview.classList.remove('hidden');
      prePlaceholder.innerText = "";
    }

    // set prediction
    predLabel.textContent = data.prediction;
    predConf.textContent = `Confidence: ${data.confidence}%`;
    predictionCard.classList.remove('hidden');
    predPlaceholder.innerHTML = "";

    // hide spinner
    loadingSpinner.classList.add('hidden');
    statusText.textContent = "Done";

  } catch (err) {
    console.error(err);
    alert('Prediction failed: ' + (err.message || 'Unknown'));
    resetUI();
  }
}
