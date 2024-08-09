import PyPDF2
import numpy as np
from qdrant_client import QdrantClient
from langchain_community.llms.ollama import Ollama
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset
import pandas as pd

# Dummy embedding function (replace with actual if available)
def get_dummy_embeddings(texts):
    return [np.random.rand(768) for _ in texts]

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def store_pdf_text_in_qdrant(qdrant_client, pdf_text):
    documents = pdf_text.split("\n\n")  # Split by paragraphs
    embeddings = get_dummy_embeddings(documents)

    for i, embedding in enumerate(embeddings):
        qdrant_client.upsert(
            collection_name="oratio",
            points=[
                {
                    "id": i,
                    "vector": embedding.tolist(),
                    "payload": {"text": documents[i]},
                }
            ]
        )

# Initialize Qdrant client
qdrant_client = QdrantClient(url='http://localhost:6333')

# Extract text from PDF and store in Qdrant
pdf_text = extract_text_from_pdf('circulaire.pdf')
store_pdf_text_in_qdrant(qdrant_client, pdf_text)

# Connect to Ollama
ollama_client = Ollama(model="mistral")

def query_rag(qdrant_client, question):
    query_embedding = get_dummy_embeddings([question])[0]
    search_result = qdrant_client.search(
        collection_name="oratio",
        query_vector=query_embedding,
        limit=5
    )
    top_texts = [hit.payload['text'] for hit in search_result]
    input_text = question + "\n" + "\n".join(top_texts)

    # Use Ollama to generate a response
    response_text = ollama_client.invoke(input_text)

    return response_text


questions = [
    "Quelle est la date de la circulaire mentionnée?",
    "Quelles sont les nouvelles directives pour la mise en œuvre?",
]

ground_truths = [
    "La circulaire est datée du 5 août 2023.",
    "Les nouvelles directives stipulent que toutes les unités doivent suivre les procédures mises à jour.",
]

answers = []
contexts = []

# Inference
for query in questions:
    answer = query_rag(qdrant_client, query)
    if not answer:
        answer = "No answer returned"  # Handle empty response
    answers.append(answer)
    
    search_result = qdrant_client.search(
        collection_name="oratio",
        query_vector=get_dummy_embeddings([query])[0],
        limit=5
    )
    context = [hit.payload['text'] for hit in search_result]
    contexts.append(context)

# Print the collected answers and contexts for debugging
print("Answers:", answers)
print("Contexts:", contexts)

# To dict
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}

# Convert dict to dataset
dataset = Dataset.from_dict(data)

# Evaluate with RAGas
evaluation_results = evaluate(
    dataset=dataset,
    llm=ollama_client,
    metrics=[ context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)
pd.set_option("display.max_colwidth", None)

df = evaluation_results.to_pandas()
df.to_csv('results.csv')
