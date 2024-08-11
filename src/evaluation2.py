import requests
import json
# Initialize Qdrant client

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range
from qdrant_client.http import models

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
def fetch_documents_with_filter(client, collection_name):
    # Assuming you have a field like 'document_id' or similar to filter on

    # Dummy vector for the purpose of completing the API call structure
    query_vector = np.random.rand(4096)  # Ensure dimensionality matches your Qdrant configuration

    # Perform the search with filtering
    hits = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=1  # Adjust based on how many documents you expect to fetch
    )

    # Extract document contents or other relevant info
    documents = []
    for hit in hits:
        documents.append(hit.payload)  # Adjust attribute access based on actual response structure

    return documents

# Example usage
documents = fetch_documents_with_filter(client, 'oratio2')
# Assuming you have these installed
from ragas import evaluate  # Update this import based on your actual library

# Sample questions and their expected ground truth answers
questions = [
    "Quelle est la date de la circulaire mentionnée?",
    "Quelles sont les nouvelles directives pour la mise en œuvre?",
]

ground_truths = [
    "La circulaire est datée du 5 août 2023.",
    "Les nouvelles directives stipulent que toutes les unités doivent suivre les procédures mises à jour.",
]

# Assume you have collected responses from OLLAMA
responses = []
for question in questions:
    prompt = f"Context: {documents}\nQuestion: {question}\nAnswer:"
    response = ollama_model.predict(prompt)
    responses.append(response)

# Create a dataset for Ragas evaluation
dataset = {
    "question": questions,
    "response": responses,
    "ground_truth": ground_truths
}

# Assuming evaluate function is ready to use such a dataset
metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
evaluation_results = evaluate(dataset, metrics=metrics)
print(evaluation_results)



