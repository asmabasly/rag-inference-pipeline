import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from PyPDF2 import PdfReader
from langchain_community.embeddings.ollama import OllamaEmbeddings
# Function to extract text from a PDF
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to chunk text into smaller parts
def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Initialize Qdrant and create collection
def initialize_qdrant():
    qdrant_client = QdrantClient(host='localhost', port=6333)

    # Check if the collection exists and create if not
    try:
        qdrant_client.get_collection(collection_name='my_collection')
        print("Collection 'my_collection' exists.")
    except Exception:
        print("Collection 'my_collection' does not exist. Creating new collection.")
        qdrant_client.recreate_collection(
            collection_name='my_collection',
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
        )
    
    return qdrant_client

# Function to store text chunks in Qdrant
def store_pdf_text_in_qdrant(qdrant_client, text_chunks, embedder, batch_size=100):
    try:
        for batch in [text_chunks[i:i + batch_size] for i in range(0, len(text_chunks), batch_size)]:
            points = [
                {
                    "id": i,
                    "vector": embedder._embed(chunk),  # Assuming embed is the correct method
                    "payload": {"text": chunk}
                }
                for i, chunk in enumerate(batch)
            ]
            qdrant_client.upsert(
                collection_name="my_collection",
                points=points
            )
        print("PDF text successfully stored in Qdrant.")
    except Exception as e:
        print(f"Error during upsert operation: {e}")

if __name__ == "__main__":
    # Path to your PDF file
    pdf_path = "circulaire.pdf"
    
    # Extract and chunk text from the PDF
    full_text = extract_text_from_pdf(pdf_path)
    text_chunks = chunk_text(full_text)

    # Initialize Qdrant client
    qdrant_client = initialize_qdrant()

    # Initialize Ollama for embeddings
    embedder = OllamaEmbeddings(model="mistral")

    # Store text chunks in Qdrant
    store_pdf_text_in_qdrant(qdrant_client, text_chunks, embedder)
