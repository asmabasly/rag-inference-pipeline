import requests
import json
# Initialize Qdrant client

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range
from qdrant_client.http import models
import pandas as pd
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall 

class OLLAMA:
    def __init__(self, model_name, api_endpoint='http://localhost:11434/api/generate', **kwargs):
        self.model_name = model_name
        self.api_endpoint = api_endpoint
        self.session = requests.Session()
        self.kwargs = {"temperature": 0.7, "n": 1, **kwargs}
        print(f"Initialized OLLAMA with model_name: {model_name}, api_endpoint: {api_endpoint}, kwargs: {self.kwargs}")

    def predict(self, question, **kwargs):
        complete_response = ""
        payload = {'model': self.model_name, 'prompt': question, **self.kwargs, **kwargs}
        with self.session.post(self.api_endpoint, json=payload, stream=True) as r:
            if r.status_code == 200:
                for line in r.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        json_response = json.loads(decoded_line)
                        complete_response += json_response.get("response", "")
                        if json_response.get("done", False):
                            break
            else:
                print(f"Error: Received status code {r.status_code}")
        return complete_response.strip()

# Example usage
ollama_model = OLLAMA(model_name="mistral")
#response = ollama_model.predict("Sample question?")
#print("Response:", response)



# Initialize the Qdrant client
client = QdrantClient(host='localhost', port=6333)

# Function to fetch document content using metadata filtering
def fetch_and_format_documents(client, collection_name):
    # Fetch documents with a dummy vector
    query_vector = np.random.rand(4096)  # Ensure dimensionality matches your Qdrant configuration
    hits = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=1  # Adjust based on the needs
    )
    
    # Extract and format document contents
    documents = []
    for hit in hits:
        # Assuming 'payload' and 'text' hold the document content; adjust as per your actual data structure
        document_text = hit.payload['text']
        documents.append(document_text.replace('\n', ' ').strip())
    
    return " ".join(documents)  # Join all documents into a single string if there are multiple

# Example usage
documents = fetch_and_format_documents(client, 'oratio2')
# Assuming you have these installed
from ragas import evaluate  # Update this import based on your actual library

from datasets import Dataset

# Example questions and their expected ground truth answers
questions = [
    "Quelle est la date de la circulaire mentionnée?",
    "Quelles sont les nouvelles directives pour la mise en œuvre?",
]
# Expected ground truth answers
ground_truths = [
    "La circulaire est datée du 24 septembre 1987.",
    "Les nouvelles directives stipulent que toutes les unités doivent suivre les procédures mises à jour."
]

responses = []
for question in questions:
    prompt = f"Context: {documents}\nQuestion: {question}\nAnswer:"
    response = ollama_model.predict(prompt)
    responses.append(response)

data = {
    "question": questions,
    "contexts":  [[documents] for _ in questions],  # Use the same flattened context for each question
    "answer": responses,
    "ground_truth": ground_truths
}

from datasets import Dataset, Features, Value, Sequence

features = Features({
    'question': Value('string'),
    'contexts': Sequence(Value('string')),  # Ensuring contexts is a sequence of strings
    'answer': Value('string'),
    'ground_truth': Sequence(Value('string'))
})

dataset = Dataset.from_dict(data)


# Run the evaluation
evaluation_results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
score_df = evaluation_results.to_pandas()
score_df.to_csv("EvaluationScores.csv", encoding="utf-8", index=False)

score_df[['faithfulness', 'answer_relevancy', 'context_precision','context_recall']].mean(axis=0)
#print("Evaluation Results:", evaluation_results)

# Assume you have collected responses from OLLAMA
