import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def clean(text):
    text = re.sub(r"[^a-zA-Z ]", " ", str(text)).lower()
    return re.sub(r"\s+", " ", text).strip()

# 1. Data load aur saaf
df = pd.read_csv("data/Resume.csv")
df["clean"] = df["Resume_str"].apply(clean)

# 2. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["Category"], test_size=0.2,
    random_state=42, stratify=df["Category"])

# 3. TF-IDF
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)

# 4. Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

results = {}
best_name, best_acc, best_model = None, 0, None

for name, m in models.items():
    print(f"Training {name}...")
    m.fit(Xtr, y_train)
    pred = m.predict(Xte)
    acc = accuracy_score(y_test, pred)
    results[name] = acc
    print(f"  Accuracy: {acc:.3f}")
    if acc > best_acc:
        best_name, best_acc, best_model = name, acc, m

print("\n===== COMPARISON =====")
for name, acc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"{name:22s} {acc*100:.1f}%")

print("\nBest model:", best_name)
best_pred = best_model.predict(Xte)
print(classification_report(y_test, best_pred))

# 5. Accuracy bar chart
plt.figure(figsize=(8, 4))
plt.bar(results.keys(), [a * 100 for a in results.values()])
plt.ylabel("Accuracy (%)")
plt.title("Model Comparison")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.close()

# 6. Confusion matrix
fig, ax = plt.subplots(figsize=(14, 12))
ConfusionMatrixDisplay.from_predictions(
    y_test, best_pred, ax=ax, xticks_rotation=90, colorbar=False)
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# 7. Save
joblib.dump(best_model, "model.pkl")
joblib.dump(tfidf, "tfidf.pkl")
print("\nSaved: model.pkl, tfidf.pkl, model_comparison.png, confusion_matrix.png")