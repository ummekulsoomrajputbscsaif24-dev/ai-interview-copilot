import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/Resume.csv")

print("Shape (rows, columns):", df.shape)
print("\nColumns:", list(df.columns))
print("\nMissing values:\n", df.isnull().sum())
print("\nCategories:\n", df["Category"].value_counts())

print("\n--- Sample resume (pehle 500 characters) ---")
print(df["Resume_str"][0][:500])

df["Category"].value_counts().plot(kind="bar", figsize=(12, 5))
plt.title("Resumes per Category")
plt.ylabel("Count")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.savefig("category_chart.png")
plt.show()