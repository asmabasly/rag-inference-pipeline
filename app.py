#1. Importation des bibliothèques et modules nécessaires

# Importation des modules FastAPI et Pydantic pour la création de l'API et la validation des données.
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langchain_qdrant import Qdrant
from langchain.chains.retrieval import create_retrieval_chain

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI, HTTPException
from ragas.metrics import faithfulness, answer_relevancy, context_recall

# Importation des modules pour la gestion asynchrone des fichiers et la configuration.
import yaml


# Importation des utilitaires personnalisés pour le téléversement et la recherche de fichiers.
from typing import Optional

# Importation de Ollama pour l'utilisation du modèle de langage et des embeddings.
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag.rag import RAG





#2. Initialisation de l'API FastAPI et configuration
app = FastAPI()

# Mount a static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Chargement de la configuration à partir du fichier YAML.
with open("config.yml", "r") as conf:
    config = yaml.safe_load(conf)
    
# Initialisation de Ollama avec la configuration chargée.
#llm = Ollama(model="mistral")
llm = Ollama(model= 'mistral', base_url = "http://localhost:11434/", verbose= True)



#3. Point d'entrée de l'API
@app.get("/")
async def read_root():
    return FileResponse("static/welcome_page.html")

#4. Téléversement de fichiers
rag = RAG(config=config, llm=llm)

@app.post("/uploadfile/")
async def upload_file(file: UploadFile):
    try:
        await rag.save_file(file)
        return {"message": f"File '{file.filename}' saved successfully"}
    except Exception as e:
        return {"error": f"An error occurred while saving the file: {str(e)}"}

#5. Définition des modèles Pydantic pour la validation des données
class Query(BaseModel):
    query: str
    similarity_top_k: Optional[int] = Field(default=1, ge=1, le=5)

class Response(BaseModel):
    search_result: str 
    source: str


#6. Recherche de fichiers
#file_searcher = rag(config=config, llm=llm)
embed_model = HuggingFaceEmbeddings(model_name=config["embedding"])

  # The qdrant client from langchain
vector_store = Qdrant(
        client=rag.qdrant_client, collection_name=config["collection_name"], 
            embeddings=embed_model,
        )


@app.post("/api/search", status_code=200)
def search(query: Query):
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2})

    result = retriever.invoke(query.query)
    
    return result 

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

@app.post("/api/chat", status_code=200)
def search(query: Query):
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": query.similarity_top_k})
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query.query})
    
    return response["answer"]

questions = ["Quelles sont les nouvelles directives pour la mise en œuvre?"]
correct_answers = ["Les nouvelles directives stipulent que toutes les unités doivent suivre les procédures mises à jour."]

# Endpoint for RAG evaluation
@app.get("/evaluate_ragas")
async def evaluate_ragas():
    try:
        question = questions[0]
        correct_answer = correct_answers[0]

        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = rag_chain.invoke({"input": question})
        generated_answer = response["answer"]
        retrieved_documents = response["retrieved_documents"]  # Assuming this is how you get the contexts

        # Calculating the metrics
        eval_faithfulness = faithfulness(generated_answer, correct_answer, retrieved_documents)
        eval_answer_relevancy = answer_relevancy(generated_answer, correct_answer)
        eval_context_recall = context_recall(retrieved_documents, correct_answer)

        return {
            "question": question,
            "generated_answer": generated_answer,
            "faithfulness": eval_faithfulness,
            "answer_relevancy": eval_answer_relevancy,
            "context_recall": eval_context_recall
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)