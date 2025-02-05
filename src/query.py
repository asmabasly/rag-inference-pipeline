from langchain_community.llms.ollama import Ollama

# Connect to Ollama using the container name
ollama_client = Ollama(model="mistral")

def query_rag(qdrant_client, question):
    from embedding import generate_embeddings

    query_embedding = generate_embeddings([question])[0]

    search_result = qdrant_client.search(
        collection_name="collection1",
        query_vector=query_embedding,
        limit=5
    )

    top_texts = [hit.payload['text'] for hit in search_result]

    input_text = question + "\n" + "\n".join(top_texts)

    # Use Ollama to generate a response
    response_text = ollama_client.invoke(input_text)

    return response_text
