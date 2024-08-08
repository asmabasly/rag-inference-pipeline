from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from langchain.chains.retrieval_qa.base import RetrievalQA

# Example questions and ground truths from your document
questions = [
    "Quelle est la date de la circulaire mentionnée?",
    "Quels sont les principaux objectifs de la circulaire?",
    "Comment les changements affecteront-ils le processus existant?",
    "Quelles sont les nouvelles directives pour la mise en œuvre?",
]

ground_truths = [
    ["La circulaire est datée du 5 août 2023."],
    ["Les objectifs principaux incluent l'amélioration de l'efficacité et la réduction des coûts."],
    ["Les changements entraîneront une simplification du processus et une amélioration de la précision."],
    ["Les nouvelles directives stipulent que toutes les unités doivent suivre les procédures mises à jour."],
]

# Placeholder for answers and contexts
answers = []
contexts = []

# Example: Using a retrieval QA chain to generate answers
# Replace this with your actual retrieval and QA chain
retrieval_qa = RetrievalQA()  # Initialize your retrieval QA instance

for query in questions:
    answer = retrieval_qa.invoke(query)
    context_docs = retrieval_qa.retriever.get_relevant_documents(query)
    context_text = [doc.page_content for doc in context_docs]

    answers.append(answer)
    contexts.append(context_text)

# Prepare data for Ragas evaluation
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truths": ground_truths
}

# Convert to dataset
dataset = Dataset.from_dict(data)

# Evaluate using Ragas
result = evaluate(
    dataset=dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)

print(result)
