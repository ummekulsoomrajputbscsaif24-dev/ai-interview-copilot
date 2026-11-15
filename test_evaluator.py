from evaluator import evaluate

keywords = ["mutable", "immutable", "ordered", "list", "tuple"]

answers = {
    "Good answer": "A list is mutable which means you can change it after creation, while a tuple is immutable and cannot be changed. Both are ordered collections in Python and keep the sequence of items. I use a list when data changes and a tuple for fixed data.",
    "Short answer": "List can change, tuple cannot.",
    "Wrong answer": "I like football and my favourite food is biryani.",
    "Empty answer": "",
}

for name, ans in answers.items():
    r = evaluate(ans, keywords)
    print(f"--- {name} ---")
    print("Score:", r["score"], "| Relevance:", r["relevance"], "| Coverage:", r["coverage"])
    print("Feedback:", " ".join(r["feedback"]))
    print()