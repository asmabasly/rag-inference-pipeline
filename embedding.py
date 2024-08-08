from langchain_community.embeddings import BedrockEmbeddings
# Uncomment the following import if using Ollama embeddings
# from langchain_community.llms.ollama import OllamaEmbeddings
import numpy as np

# Dummy embedding function (replace with actual if available)
def get_dummy_embeddings(texts):
    return [np.random.rand(768) for _ in texts]

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
