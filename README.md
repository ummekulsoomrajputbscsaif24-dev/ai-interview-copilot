# ai-interview-copilot
# AI Interview Copilot

Resume-based adaptive mock interview system (Semester 5 ML project).

## Features
- Resume category classification (TF-IDF + Random Forest, 70.6% accuracy)
- Role-specific question selection
- Answer evaluation using TF-IDF + cosine similarity
- Voice input and output (Web Speech API)
- SQLite database

## Setup
1. Install Python 3.11
2. `py -3.11 -m pip install flask pandas numpy scikit-learn nltk matplotlib seaborn joblib pdfplumber`
3. Download Resume.csv from Kaggle: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset and put it in `data/`
4. Run `py -3.11 app.py` and open http://127.0.0.1:5000

## Team
- Ummekulsoom (Team Lead): ML model, backend
- Kubra Shams: Data preprocessing, frontend, database