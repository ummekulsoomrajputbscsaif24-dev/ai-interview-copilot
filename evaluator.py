import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FILLERS = ["um", "uh", "like", "basically", "actually", "you know"]


def _words(text):
    return re.findall(r"[a-z]+", text.lower())


def relevance_score(answer, keywords):
    """TF-IDF + cosine similarity: answer vs expected keywords."""
    reference = " ".join(keywords)
    try:
        vec = TfidfVectorizer().fit([answer, reference])
        a = vec.transform([answer])
        r = vec.transform([reference])
        return float(cosine_similarity(a, r)[0][0])
    except ValueError:
        return 0.0  # empty answer


def keyword_coverage(answer, keywords):
    """Which keywords were found and which were missed."""
    words = set(_words(answer))
    found = []
    missed = []
    for k in keywords:
        if k.lower() in words or k.lower() + "s" in words or k.lower().rstrip("s") in words:
            found.append(k)
        else:
            missed.append(k)
    coverage = len(found) / len(keywords) if keywords else 0
    return coverage, found, missed


def length_score(answer):
    """40+ words = complete answer."""
    n = len(_words(answer))
    if n >= 40:
        return 1.0
    return n / 40


def count_fillers(answer):
    text = " " + " ".join(_words(answer)) + " "
    return sum(text.count(" " + f + " ") for f in FILLERS)


def evaluate(answer, keywords):
    """Returns the answer score (0-100) and English feedback."""
    answer = (answer or "").strip()
    n_words = len(_words(answer))

    if n_words == 0:
        return {
            "score": 0,
            "relevance": 0,
            "coverage": 0,
            "found": [],
            "missed": list(keywords),
            "feedback": ["No answer given. Try to speak or type at least 2-3 sentences."],
        }

    rel = relevance_score(answer, keywords)
    cov, found, missed = keyword_coverage(answer, keywords)
    length = length_score(answer)

    score = round((0.4 * rel + 0.4 * cov + 0.2 * length) * 100)

    feedback = []
    if missed:
        feedback.append("You missed these key points: " + ", ".join(missed) + ".")
    if n_words < 20:
        feedback.append("Your answer is too short. Add an example or more detail.")
    elif n_words < 40:
        feedback.append("Good start, but try to explain a little more.")
    fillers = count_fillers(answer)
    if fillers >= 3:
        feedback.append(f"You used filler words {fillers} times. Try to speak more confidently.")
    if score >= 75:
        feedback.insert(0, "Excellent answer!")
    elif score >= 50:
        feedback.insert(0, "Good answer, but it can be improved.")
    else:
        feedback.insert(0, "Needs improvement.")

    return {
        "score": score,
        "relevance": round(rel * 100),
        "coverage": round(cov * 100),
        "found": found,
        "missed": missed,
        "feedback": feedback,
    }