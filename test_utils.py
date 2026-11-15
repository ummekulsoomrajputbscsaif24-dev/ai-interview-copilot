
from utils import predict_category, extract_skills, get_questions

sample = """
Experienced software developer with strong skills in Python, Flask, SQL and
JavaScript. Built web applications, worked with Git, and used machine learning
for data analysis. Good communication and teamwork.
"""

category = predict_category(sample)
skills = extract_skills(sample)
print("Category:", category)
print("Skills:", skills)
print("\nQuestions:")
for i, q in enumerate(get_questions(category, skills), 1):
    print(f"{i}. {q['q']}")