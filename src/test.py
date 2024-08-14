import pdfplumber
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.qdrant import Qdrant
from langchain.chains.retrieval_qa.base import RetrievalQA
from transformers import pipeline

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        all_text = ''
        for page in pdf.pages:
            all_text += page.extract_text()
    return all_text

pdf_path = r'C:\Users\Tifa\Desktop\rag\src\data\circulaire.pdf'
pdf_text = extract_text_from_pdf(pdf_path)
print(pdf_text)


# Assuming you are using LLaMA or a suitable transformer model
#llama_model = pipeline("text-generation", model="allenai/llama_7b")  # Adjust model as per your requirement
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
#llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
# Load model directly

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
llama_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")
# Setup embeddings
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})

# Generate embeddings for extracted text
pdf_embeddings = embeddings.encode([pdf_text])  # Assuming the whole PDF is one document

# Initialize Qdrant
doc_store = Qdrant.from_texts([pdf_text], [pdf_embeddings], location=":memory:", prefer_grpc=True, collection="pdf_collection")

# Setup the RetrievalQA with LLaMA
qa = RetrievalQA(llm=llama_model, retriever=doc_store.as_retriever(search_kwargs={"k": 5}))

# Query processing
question = "Quelles sont les nouvelles directives pour la mise en œuvre?"
result = qa(question)
print(result)
