from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from fastapi import UploadFile, HTTPException, File
import os 
import re
import aiofiles
from qdrant_client import QdrantClient
from qdrant_client.http import models

class RAG:
    def __init__(self,config,llm):
        self.llm = llm
        self.config = config
        self.qdrant_client = QdrantClient(
            url=self.config['qdrant_local'],
            #timeout=self.config["qdrant_timeout"]
        )

      
        self.qdrant_client.recreate_collection(
            collection_name=self.config['collection_name'],
            vectors_config=models.VectorParams(
            size=768,
            distance=models.Distance.COSINE
            )
        )
    async def save_pdf(self, file: UploadFile, file_path: str):
        async with aiofiles.open(file_path, "wb") as saved_file:
            await saved_file.write(await file.read())

    async def save_csv(self, file: UploadFile, file_path: str):

        async with aiofiles.open(file_path, "wb") as saved_file:
            await saved_file.write(await file.read())

    async def save_text(self, file: UploadFile, file_path: str):

        async with aiofiles.open(file_path, "wb") as saved_file:
            await saved_file.write(await file.read())

    async def save_json(self, file: UploadFile, file_path: str):

        async with aiofiles.open(file_path, "wb") as saved_file:
            await saved_file.write(await file.read())


    def remove_non_alphanumeric(input_string):
        # Replace non-alphanumeric characters with an empty string
        cleaned_string = re.sub(r'[^a-zA-Z0-9]', '', input_string)
        return cleaned_string
    

    async def save_file(self, file: UploadFile):
        """
        Save the uploaded file based on its type.
        """
        file_path = os.path.join(self.config['upload_dir'], file.filename)
        file_extension = os.path.splitext(file.filename)[1].lower()

        if file_extension == ".pdf":
            await self.save_pdf(file, file_path)
        elif file_extension == ".csv":
            await self.save_csv(file, file_path)
        elif file_extension == ".txt":
            await self.save_text(file, file_path)
        elif file_extension == ".json":
            await self.save_json(file, file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        print("Loading Embedder...")
        data = RAG(self.config,self.llm)
        
        print("Embedder Loaded",)
        embed_model = HuggingFaceEmbeddings(model_name=self.config["embedding"])
    
        
        
        
        print("Embedder modelled")
 
    
        await data.ingest(embedder=embed_model)

    async def ingest(self, embedder):
        print("Indexing data...")
        loader = PyPDFLoader("./rag/reg-bancaire.pdf")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        qdrant = Qdrant.from_documents(
            docs,
            embedder,
            url=self.config['qdrant_local'],
            collection_name=self.config['collection_name'],
        )
        
        print(
            f"Data indexed successfully to Qdrant. Collection: {self.config['collection_name']}"
        )
        return qdrant