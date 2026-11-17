import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

STOP = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def preprocess(text):
    """Cleans text: lowercase, remove symbols, remove stopwords, lemmatize."""
    text = re.sub(r"[^a-zA-Z ]", " ", str(text)).lower()
    tokens = [LEMMATIZER.lemmatize(w) for w in text.split()
              if w not in STOP and len(w) > 2]
    return " ".join(tokens)


if __name__ == "__main__":
    sample = "I was managing projects and managed teams, using Python and databases!"
    print("Before:", sample)
    print("After :", preprocess(sample))