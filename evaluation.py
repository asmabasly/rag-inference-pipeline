from datasets import Dataset
from ragas import evaluate
from langchain.chains.retrieval import create_retrieval_chain
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from query import query_rag
from embedding import get_dummy_embeddings
from qdrant_setup import initialize_qdrant

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

qdrant_client = initialize_qdrant()
for query in questions:
    answer = query_rag(qdrant_client, query)
    answers.append(answer)
    
    search_result = qdrant_client.search(
        collection_name="oratio",
        query_vector=get_dummy_embeddings([query])[0],
        limit=5
    )
    context = [hit.payload['text'] for hit in search_result]
    contexts.append(context)

# Example: Using a retrieval QA chain to generate answers
# Replace this with your actual retrieval and QA chain
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
