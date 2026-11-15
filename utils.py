import re
import json
import random
import joblib
import pdfplumber

model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

with open("questions.json", encoding="utf-8") as f:
    QUESTIONS = json.load(f)

SKILLS = [
    "python", "java", "javascript", "html", "css", "sql", "mysql", "flask", "django",
    "react", "machine learning", "data analysis", "excel", "power bi", "tableau",
    "communication", "leadership", "teamwork", "problem solving", "time management",
    "project management", "recruitment", "payroll", "accounting", "tally", "quickbooks",
    "auditing", "taxation", "budgeting", "sales", "marketing", "negotiation",
    "customer service", "lesson planning", "classroom management", "autocad",
    "solidworks", "networking", "linux", "git", "photoshop", "seo", "research",
    "presentation", "teaching", "training", "social media",
]

def clean(text):
    text = re.sub(r"[^a-zA-Z ]", " ", str(text)).lower()
    return re.sub(r"\s+", " ", text).strip()

def read_pdf(file):
    """PDF file se text nikalta hai."""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + " "
    return text

def predict_category(text):
    vec = tfidf.transform([clean(text)])
    return model.predict(vec)[0]

def extract_skills(text):
    text = text.lower()
    found = []
    for skill in SKILLS:
        if re.search(r"\b" + re.escape(skill) + r"\b", text):
            found.append(skill)
    return found

def get_questions(category, skills, total=6):
    result = list(QUESTIONS["INTRO"])

    # Category ke sawal
    pool = list(QUESTIONS.get(category, QUESTIONS["GENERAL"]))
    random.shuffle(pool)
    result += pool[:3]

    # Resume ki skills par sawal
    for skill in skills[:2]:
        result.append({
            "q": f"You mention {skill} in your resume. Tell me about a project or task where you used it.",
            "keywords": [skill, "project", "used", "result", "learned"],
        })

    # Kam pad jayein to general sawal
    if len(result) < total:
        extra = [q for q in QUESTIONS["GENERAL"] if q not in result]
        result += extra[: total - len(result)]

    return result[:total]