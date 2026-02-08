BioPrint – Identifying Blood Group Type Through Fingerprint Images
Overview

BioPrint is a Final Year Project that predicts a person’s blood group using fingerprint images.
It uses deep learning on fingerprint patterns and provides results through a web interface.

The system has:

a backend API (FastAPI + PyTorch)

a frontend website (HTML, CSS, JavaScript)

Blood Groups Supported

The model predicts 8 blood groups:

A+, A-

B+, B-

AB+, AB-

O+, O-

BIOPRINT/
│
├── backend/
│   ├── main.py            # FastAPI server
│   ├── inference.py       # Model loading + prediction logic
│   ├── requirements.txt
│   ├── model/
│   │   └── efficientnet_b0.pth
│   └── media/             # Saved images (auto-created)
│
├── frontend/
│   ├── index.html
│   ├── about.html
│   ├── research.html
│   ├── style.css
│   └── script.js
│

How the System Works (Simple Flow)

User uploads a fingerprint image from the frontend

Image is sent to the backend /predict API

Image preprocessing is applied:

grayscale

resize to 224×224

CLAHE (contrast enhancement)

blur and sharpening

Image is passed to EfficientNet-B0

Model predicts blood group and confidence

Backend returns:

predicted blood group

confidence percentage

original image

preprocessed image

Frontend shows the result visually

Model Details

Architecture: EfficientNet-B0

Framework: PyTorch

Input size: 224 × 224

Classes: 8 blood groups

Device: CPU or GPU (auto-detected)

The model is loaded once at startup for fast predictions.

Backend (FastAPI)
Start Backend

From the backend folder:

python main.py


The API runs at:

http://127.0.0.1:8000

API Endpoints

GET /
Checks if API is running

POST /predict
Upload fingerprint image and get prediction

Frontend

The frontend is a simple static website.

How it works

Drag & drop or browse fingerprint image

Shows:

original image

preprocessed image

predicted blood group

confidence

Frontend communicates with backend using JavaScript Fetch API.

Technologies Used
Backend

Python

FastAPI

PyTorch

Torchvision

OpenCV

PIL

Uvicorn

Frontend

HTML

CSS

JavaScript

Important Notes

Model file efficientnet_b0.pth must be placed inside:

backend/model/


The media folder is created automatically

.gitignore is used to avoid uploading:

virtual environment

cache files

large model artifacts (if needed)

Project Purpose

This project explores biometric identification using fingerprints to predict blood group types, providing a non-invasive and automated approach for research and academic study.

Author

Ahmad Sohaib
Final Year Project – Data Science

If you want next, I can:

simplify this README even more

make it IEEE / university format

add API examples

help you write project abstract
├── .gitignore
└── README.md
