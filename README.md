**BioType: Identifying Blood Group Type Through Fingerprint Images
Overview**

BioPrint is a Final Year Project that predicts a person’s blood group using fingerprint images.
It uses deep learning and shows results through a web application.

The system has:

a backend API built with FastAPI and PyTorch

a frontend website built with HTML, CSS, and JavaScript

Blood Groups Supported

The model predicts 8 blood groups:

A+, A−

B+, B−

AB+, AB−

O+, O−

**Project Structure**
BIOPRINT/
│
├── backend/
│   ├── main.py              # FastAPI server
│   ├── inference.py         # Model loading and prediction logic
│   ├── requirements.txt
│   ├── model/
│   │   └── efficientnet_b0.pth
│   └── media/               # Saved images (auto-created)
│
├── frontend/
│   ├── index.html
│   ├── about.html
│   ├── research.html
│   ├── style.css
│   └── script.js
│
├── .gitignore
└── README.md

**How the System Works (Simple Flow)**

1. User uploads a fingerprint image from the frontend
2. Image is sent to the backend /predict API
3. Image preprocessing is applied:
3.1. convert to grayscale
3.2. resize to 224 × 224
3.3. apply CLAHE for better contrast
3.4. apply blur and sharpening
4. Image is passed to EfficientNet-B0
5. Model predicts blood group and confidence
6. Backend sends back:
6.1. predicted blood group
6.2. confidence percentage
6.3. original image
6.4. preprocessed image
7. Frontend displays the result clearly

**Model Details**

1. Architecture: EfficientNet-B0
2. Framework: PyTorch
3. Input Size: 224 × 224
4. Classes: 8 blood groups
5. Device: CPU or GPU (auto detected)
The model is loaded once at startup to make predictions faster.

**Backend (FastAPI)**
Start Backend

Run this command from the backend folder:
python main.py

The API will run at:
http://127.0.0.1:8000

**Frontend**
The frontend is a simple static website.
How it Works
**User drags or selects a fingerprint image**
The page shows:
1. original image
2. preprocessed image
3. predicted blood group
4. confidence percentage
Frontend communicates with backend using the JavaScript Fetch API.

**Technologies Used
Backend**
1. Python
2. FastAPI
3. PyTorch
4. Torchvision
5. OpenCV
6. PIL
7. Uvicorn

**Frontend**
1. HTML
2. CSS
3. JavaScript

**Important Notes**
The model file efficientnet_b0.pth must be placed inside:
backend/model/

The media folder is created automatically

.gitignore is used to avoid uploading:
1. virtual environment
2. cache files
3. large unnecessary files

**Project Purpose**
This project studies biometric identification by using fingerprints to predict blood group types.
It provides a non-invasive, automated, and research-focused solution.
