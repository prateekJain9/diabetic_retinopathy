# RetinaAI — Diabetic Retinopathy Flask Web App

A full-stack Flask web application for detecting Diabetic Retinopathy using
a MobileNetV2 CNN trained via Federated Learning (FedAvg).

## Features
- Home page with project overview
- User registration & login (SQLite, SHA-256 password hashing)
- **Predict page** — upload retinal fundus images, get DR/No_DR result with confidence
- **Analytics page** — diagnosis history, bar chart, donut chart, full table
- Session management, flash messages, responsive dark UI

## Project Structure

```
dr_webapp/
├── app.py                  # Main Flask application
├── requirements.txt
├── dr_fedavg_model.h5      # ← Place your model here
├── instance/
│   └── dr_app.db           # SQLite database (auto-created)
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── uploads/            # Uploaded images (auto-created)
└── templates/
    ├── base.html
    ├── home.html
    ├── login.html
    ├── register.html
    ├── dashboard.html
    ├── predict.html
    └── analytics.html
```

## Setup & Run

```bash
# 1. Copy your model into the project root
cp /path/to/dr_fedavg_model.h5 dr_webapp/

# 2. Install dependencies
cd dr_webapp
pip install -r requirements.txt

# 3. Run the app
python app.py
```

Open http://localhost:5000 in your browser.

## Model Details
| Property | Value |
|---|---|
| Backbone | MobileNetV2 (ImageNet weights) |
| Algorithm | FedAvg Federated Learning |
| Input size | 224×224×3 |
| Classes | DR (Diabetic Retinopathy) / No_DR (Healthy) |
| Output | Softmax probabilities |

## Notes
- The model file `dr_fedavg_model.h5` must be placed in the same folder as `app.py`
- Upload formats supported: JPG, JPEG, PNG, BMP, TIFF (max 16 MB)
- This app is for **clinical assistance only** — not a substitute for professional diagnosis
