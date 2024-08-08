from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

def initialize_qdrant():
    try:
        qdrant_client = QdrantClient(host="localhost", port=6333)
        collection_name = "my_collection"

        if not qdrant_client.collection_exists(collection_name):
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=128, distance=Distance.COSINE)
            )
            print(f"Collection '{collection_name}' created.")
        else:
            print(f"Collection '{collection_name}' exists.")
        
        return qdrant_client

    except Exception as e:
        print(f"Error initializing Qdrant client: {e}")
        return None

qdrant_client = initialize_qdrant()