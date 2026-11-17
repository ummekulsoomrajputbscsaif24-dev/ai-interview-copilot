import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from preprocess import preprocess
import re

def basic_clean(text):
    text = re.sub(r"[^a-zA-Z ]", " ", str(text)).lower()
    return re.sub(r"\s+", " ", text).strip()

df = pd.read_csv("data/Resume.csv")
print("Processing... (1-2 minutes)")
df["basic"] = df["Resume_str"].apply(basic_clean)
df["nltk"] = df["Resume_str"].apply(preprocess)

for col in ["basic", "nltk"]:
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[col], df["Category"], test_size=0.2, random_state=42, stratify=df["Category"])
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    model = LinearSVC().fit(tfidf.fit_transform(X_tr), y_tr)
    acc = accuracy_score(y_te, model.predict(tfidf.transform(X_te)))
    print(f"{col:6s} preprocessing -> accuracy: {acc*100:.1f}%")