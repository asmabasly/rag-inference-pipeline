from ragas.langchain.evalchain import RagasEvaluatorChain
from qdrant_client import QdrantClient
#from langchain_community.llms.ollama import Ollama
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas import evaluate

class OLLAMA:
    def __init__(self, model_name, api_endpoint='http://localhost:11434/api/generate', **kwargs):
        self.model_name = model_name
        self.api_endpoint = api_endpoint
        self.session = requests.Session()
        self.kwargs = {"temperature": 0.7, "n": 1, **kwargs}
        print(f"Initialized OLLAMA with model_name: {model_name}, api_endpoint: {api_endpoint}, kwargs: {self.kwargs}")

    def predict(self, question, **kwargs):
        output = ""
        payload = {'model': self.model_name, 'prompt': question, **self.kwargs, **kwargs}
        with self.session.post(self.api_endpoint, json=payload, stream=True) as r:
            if r.status_code == 200:
                for line in r.iter_lines():
                    if line:
                        j = json.loads(line.decode('utf-8'))
                        output += j.get("response", "")
                        if j.get("done", True):
                            break
            else:
                print(f"Error: Received status code {r.status_code}")
        return output.strip()

    def __call__(self, question, **kwargs):
        return self.predict(question, **kwargs)

# Initialize OLLAMA with Mistral model
mistral_model = OLLAMA(model_name="mistral")

# Initialize your OLLAMA model
#ollama_model = Ollama(model_name="mistral", api_endpoint="http://localhost:11434/api/generate")

# Connect to your Qdrant instance
qdrant_client = QdrantClient(host="localhost", port=6333)

# Setup the evaluator chain with OLLAMA and Qdrant
evaluator_chain = RagasEvaluatorChain(
    llm=mistral_model,
    embeddings_db=qdrant_client
)

def fetch_data_from_qdrant(collection_name):
    # Fetch data from a Qdrant collection
    # This is a simplified example. You might need to paginate or use filters based on your setup.
    response = qdrant_client.get(collection_name, output_fields=["text"])
    return [item['text'] for item in response['result']['hits']]

def fetch_embeddings_from_qdrant(client, collection_name, document_ids):
    embeddings = []
    for doc_id in document_ids:
        response = client.get(
            collection_name=collection_name,
            document_id=doc_id,
            output_fields=["embedding"]
        )
        embeddings.append(response['result']['hits'][0]['embedding'])
    return embeddings

# Assuming your collection is named 'my_collection'
dataset = fetch_data_from_qdrant('oratio2')
# Format dataset if necessary
formatted_dataset = [{'prompt': text, 'expected_response': 'some_expected_response'} for text in dataset]

metrics = [context_precision, faithfulness]

# Assuming embeddings are stored and we are ready to evaluate
evaluation_embeddings = fetch_embeddings_from_qdrant(qdrant_client, 'oratio2', range(len(dataset)))

# Run the evaluation, assuming Ragas can use these embeddings directly
results = evaluate(
    dataset=formatted_dataset,
    llm=mistral_model,
    embeddings=evaluation_embeddings,  # Pass embeddings directly if supported
    metrics=metrics
)

print("Evaluation Results:", results)
