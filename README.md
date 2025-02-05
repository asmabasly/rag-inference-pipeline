# **RAG Inference Pipeline**
A **Retrieval-Augmented Generation (RAG)** pipeline that integrates **Qdrant** for vector search and **Ollama** for LLM inference, allowing users to query extracted and processed text from PDF documents.

---

## 📌 **Table of Contents**
- [Installation](#-installation)
- [Modifying Paths and Queries](#-modifying-paths-and-queries)
- [Running the Pipeline](#-running-the-pipeline)

---

## 🚀 **Installation**
To set up the environment, you need **Poetry**, **Docker**, **Qdrant**, and **Ollama**.

### **1️⃣ Install Docker**
Ensure you have Docker installed:

- **Linux/macOS**:  
  ```bash
  curl -fsSL https://get.docker.com | sh

-  **Windows**: 
Download and install from [Docker Desktop](https://docs.docker.com/engine/install/)

### **2️⃣ Install Poetry**
Poetry is used to manage dependencies. Install it with:
```bash
    pip install poetry
    poetry install
```
Activate the Poetry virtual environment:
 ```bash
    poetry shell
```

### **3️⃣ Install Ollama**
Ollama is required for local LLM inference. Install it from:

- **Linux/macOS:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

- **Windows:**
Download and install from [Ollama's website](https://ollama.com/download)

Ollama is used to generate answers from retrieved data. the model used is Mistral:
  ```bash
    ollama pull mistral
   ```

## **Modifying Paths & Queries**
Change this path to where your PDFs in `cleaningText.py`
```bash
pdf_path = 'put/your/path/here/data.pdf'
```
 
To query the RAG system, change this line in `main.py:`
```bash
question = ''
```

## **🔥 Running the Pipeline**
Once everything is installed, follow these steps:
 - Start Qdrant (Vector Database)
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant 
```
- Run the Main Pipeline
```bash
poetry run python main.py
```
This will:

- Initialize Qdrant (ensure it's running).
- Extract text from PDFs and store it in Qdrant.
- Process and clean the extracted text.
- Perform a query using Mistral.