# Phishing Detector

A machine learning tool that detects phishing attempts by analyzing URLs and text content. Built with Python and FastAPI.

## Features

- 🔗 URL analysis — flags suspicious links based on structural features
- 📝 Text analysis — scans message/email content for phishing patterns
- 🖼️ Image analysis — detects phishing indicators in screenshots (OCR-based)
- 🤖 ML model trained with scikit-learn (saved as `phishing_model.joblib`)
- ⚡ REST API built with FastAPI

## Tech Stack

- **Language:** Python 3.x
- **Web framework:** FastAPI + Uvicorn
- **ML:** scikit-learn, NumPy, joblib
- **NLP/URL parsing:** tldextract, httpx
- **Testing:** pytest

## Project Structure
Phishing Detector/
├── main.py # FastAPI entry point
├── url_analyzer.py # URL feature extraction & analysis
├── text_analyzer.py # Text-based phishing detection
├── image_analyzer.py # Image-based detection
├── models.py # Pydantic models / ML model wrappers
├── phishing_model.joblib # Trained ML model
├── requirements.txt # Python dependencies
├── tests/ # Unit tests
└── README.md


## Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/akremmohamed/Phishing-Detector.git
   cd Phishing-Detector
Create and activate a virtual environment

bash
python3 -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows
Install dependencies

bash
pip install -r requirements.txt
Usage
Run the API
bash
uvicorn main:app --reload
Then open http://127.0.0.1:8000/docs for the interactive API docs (Swagger UI).

Example request
bash
curl -X POST http://127.0.0.1:8000/analyze-url \
  -H "Content-Type: application/json" \
  -d '{"url": "http://suspicious-site.tk/login"}'
Running Tests
bash
pytest tests/
How It Works
Input (URL, text, or image) is normalized.

Feature extractors produce a vector of relevant signals.

The trained model (phishing_model.joblib) scores the input.

A verdict (phishing / safe) and confidence score are returned.

Author
Akrem Mohamed — github.com/akremmohamed

License
MIT (or specify your preferred license)
