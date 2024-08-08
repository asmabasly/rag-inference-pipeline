import requests

import requests

def test_qdrant_connection():
    try:
        response = requests.get("http://172.18.0.3:6333/collections", timeout=10)
        print("Qdrant Status Code:", response.status_code)
        print("Qdrant Response:", response.json())
    except requests.exceptions.RequestException as e:
        print("Error connecting to Qdrant:", e)

def test_ollama_connection():
    try:
        response = requests.get("http://172.18.0.2:11434/api/your-endpoint",timeout=10)  # Replace with actual endpoint
        print("Ollama Status Code:", response.status_code)
        print("Ollama Response:", response.json())
    except requests.exceptions.RequestException as e:
        print("Error connecting to Ollama:", e)

if __name__ == "__main__":
    test_qdrant_connection()
    test_ollama_connection()
