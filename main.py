from qdrant_setup import initialize_qdrant
from pdf_processing import extract_text_from_pdf
from embedding import store_pdf_text_in_qdrant
from query import query_rag
import socket

def main():
    # Initialize Qdrant (ensure it is running separately)
    qdrant_client = initialize_qdrant()

    # Extract and store PDF text
    pdf_text = extract_text_from_pdf("circulaire.pdf")
    store_pdf_text_in_qdrant(qdrant_client, pdf_text)

    # Example query
    question = " Quel est le délai pour déclarer les avoirs étrangers pour les Tunisiens ayant changé de résidence ?"
    response = query_rag(qdrant_client, question)
    print("Response:", response)

if __name__ == "__main__":
    main()
