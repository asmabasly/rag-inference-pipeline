from langchain_community.embeddings import BedrockEmbeddings
# Uncomment the following import if using Ollama embeddings
# from langchain_community.llms.ollama import OllamaEmbeddings
import numpy as np

# Dummy embedding function (replace with actual if available)
def get_dummy_embeddings(texts):
    return [np.random.rand(128) for _ in texts]

def store_pdf_text_in_qdrant(qdrant_client, pdf_text, batch_size=100):
    if qdrant_client is None:
        print("Qdrant client is not initialized.")
        return

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    try:
        for batch in chunks(pdf_text, batch_size):
            points = [
                {
                    "id": i,
                    "vector": get_dummy_embeddings(text),
                    "payload": {"text": text}
                }
                for i, text in enumerate(batch)
            ]
            qdrant_client.upsert(
                collection_name="my_collection",
                points=points
            )
        print("PDF text successfully stored in Qdrant.")
    except Exception as e:
        print(f"Error during upsert operation: {e}")

