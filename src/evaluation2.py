from ragas.langchain.evalchain import RagasEvaluatorChain
from qdrant_client import QdrantClient
from langchain_community.llms.ollama import Ollama

# Initialize your OLLAMA model
ollama_model = Ollama(model_name="mistral", api_endpoint="http://localhost:11434/api/generate")

# Connect to your Qdrant instance
qdrant_client = QdrantClient(host="localhost", port=6333)

# Setup the evaluator chain with OLLAMA and Qdrant
evaluator_chain = RagasEvaluatorChain(
    llm=ollama_model,
    embeddings_db=qdrant_client
)
