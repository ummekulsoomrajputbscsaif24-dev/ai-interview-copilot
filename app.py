import sqlite3
import uuid
from flask import Flask, render_template, request, redirect, url_for, session, jsonify

from utils import read_pdf, predict_category, extract_skills, get_questions
from evaluator import evaluate

app = Flask(__name__)
app.secret_key = "change-this-secret"
DB = "interview.db"


def db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = db()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        category TEXT,
        skills TEXT,
        created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS answers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        question TEXT,
        answer TEXT,
        score INTEGER,
        feedback TEXT
    );
    """)
    conn.commit()
    conn.close()


# In-memory storage of questions per session (simple, fine for a demo)
SESSION_QUESTIONS = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    text = request.form.get("resume_text", "").strip()
    file = request.files.get("resume_file")

    if file and file.filename.lower().endswith(".pdf"):
        text = read_pdf(file)

    if len(text) < 50:
        return render_template("index.html", error="Please upload a PDF or paste at least a few lines of your resume.")

    category = predict_category(text)
    skills = extract_skills(text)
    questions = get_questions(category, skills)

    sid = str(uuid.uuid4())
    SESSION_QUESTIONS[sid] = questions
    session["sid"] = sid

    conn = db()
    conn.execute("INSERT INTO sessions (id, category, skills) VALUES (?, ?, ?)",
                 (sid, category, ", ".join(skills)))
    conn.commit()
    conn.close()

    return redirect(url_for("interview"))


@app.route("/interview")
def interview():
    sid = session.get("sid")
    if not sid or sid not in SESSION_QUESTIONS:
        return redirect(url_for("index"))
    conn = db()
    row = conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    conn.close()
    questions = [q["q"] for q in SESSION_QUESTIONS[sid]]
    return render_template("interview.html", questions=questions,
                           category=row["category"], skills=row["skills"])


@app.route("/answer", methods=["POST"])
def answer():
    sid = session.get("sid")
    data = request.get_json()
    idx = int(data["index"])
    text = data.get("answer", "")

    q = SESSION_QUESTIONS[sid][idx]
    result = evaluate(text, q["keywords"])

    conn = db()
    conn.execute("INSERT INTO answers (session_id, question, answer, score, feedback) VALUES (?,?,?,?,?)",
                 (sid, q["q"], text, result["score"], " ".join(result["feedback"])))
    conn.commit()
    conn.close()
    return jsonify(result)


@app.route("/result")
def result():
    sid = session.get("sid")
    if not sid:
        return redirect(url_for("index"))
    conn = db()
    rows = conn.execute("SELECT * FROM answers WHERE session_id=? ORDER BY id", (sid,)).fetchall()
    info = conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    conn.close()
    avg = round(sum(r["score"] for r in rows) / len(rows)) if rows else 0
    return render_template("result.html", rows=rows, avg=avg, category=info["category"])


if __name__ == "__main__":
    init_db()
    app.run(debug=True)