# **RAG Inference Pipeline**
A **Retrieval-Augmented Generation (RAG)** pipeline that integrates **Qdrant** for vector search and **Ollama** for LLM inference, allowing users to query extracted and processed text from PDF documents.

---

## 📌 **Table of Contents**
- [Installation](#installation)
- [Running the Pipeline](#running-the-pipeline)
- [Qdrant Setup](#qdrant-setup)
- [Ollama Setup](#ollama-setup)
- [Modifying Paths & Queries](#modifying-paths--queries)
- [Example Commands](#example-commands)
- [Contributing](#contributing)
- [License](#license)

---

## 🚀 **Installation**
To set up the environment, you need **Poetry**, **Docker**, **Qdrant**, and **Ollama**.

### **1️⃣ Install Docker**
Ensure you have Docker installed:

- **Linux/macOS**:  
  ```bash
  curl -fsSL https://get.docker.com | sh

-  **Windows**: 
Download and install from Docker Desktop.

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


  

