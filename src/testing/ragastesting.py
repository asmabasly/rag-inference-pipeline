from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms.ollama import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from ragas import evaluate
from datasets import Dataset

from tqdm.notebook import tqdm
from langchain_huggingface import HuggingFaceEmbeddings

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

import nest_asyncio

def create_chunks_from_pdf(data_path, chunk_size, chunk_overlap):

   '''
   This function takes a directory of PDF files and creates chunks of text from each file.
   The text is split into chunks of size `chunk_size` with an overlap of `chunk_overlap`.
   This chunk is then converted into a langchain Document object.

   Args:
      data_path (str): The path to the directory containing the PDF files.
      chunk_size (int): The size of each chunk.
      chunk_overlap (int): The overlap between each chunk.

   Returns:
      docs (list): A list of langchain Document objects, each containing a chunk of text.
   '''

   # Load the documents from the directory
   loader = DirectoryLoader(data_path, loader_cls=PyPDFLoader)

   # Split the documents into chunks
   text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size,
      chunk_overlap=chunk_overlap,
      length_function=len,
      is_separator_regex=False,
   )
   docs = loader.load_and_split(text_splitter=text_splitter)
   return docs


def index_documents_and_retrieve(docs, embeddings):

    '''
    This function uses the Qdrant library to index the documents using the chunked text and embeddings model.
    For the simplicity of the example, we are using in-memory storage only.

    Args:
    docs: List of documents generated from the document loader of langchain
    embeddings: List of embeddings generated from the embeddings model

    Returns:
    retriever: Qdrant retriever object which can be used to retrieve the relevant documents
    '''

    qdrant = Qdrant.from_documents(
        docs,
        embeddings,
        location=":memory:",  # Local mode with in-memory storage only
        collection_name="my_collection",
    )

    retriever = qdrant.as_retriever()

    return retriever


def build_rag_chain(llm, retriever):

    '''
    This function builds the RAG chain using the LLM model and the retriever object. 
    The RAG chain is built using the following steps:
    1. Retrieve the relevant documents using the retriever object
    2. Pass the retrieved documents to the LLM model along with prompt generated using the context and question
    3. Parse the output of the LLM model

    Args:
    llm: LLM model object
    retriever: Qdrant retriever object

    Returns:
    rag_chain: RAG chain object which can be used to answer the questions based on the context
    '''
    
    template = """
        Answer the question based only on the following context:
        
        {context}
        
        Question: {question}
        """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context","question"]
        )
    
    rag_chain = (
        {"context": retriever,  "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

data_path = r'C:\Users\Tifa\Desktop\rag\data'
chunk_size = 500
chunk_overlap = 50
docs = create_chunks_from_pdf(data_path, chunk_size, chunk_overlap)
#embedding_models = ['BAAI/bge-large-en']
# Load the embeddings model
#embeddings = HuggingFaceEmbeddings(model_name=embedding_models[0])
embedder = OllamaEmbeddings(model="mistral")

retriever = index_documents_and_retrieve(docs, embedder)
llm = Ollama(model="mistral")
rag_chain = build_rag_chain(llm, retriever)
rag_chain.invoke('What is this document about?')

questions = [
    "Quelle est la date de la circulaire mentionnée?",
    "Quelles sont les nouvelles directives pour la mise en œuvre?",
]

ground_truths = [
    "La circulaire est datée du 5 août 2023.",
    "Les nouvelles directives stipulent que toutes les unités doivent suivre les procédures mises à jour.",
]

def create_test_case(questions, ground_truth, rag_chain, retriever):
    '''
    This function creates a test case for the RAG model
    It takes a list of questions and the corresponding ground truth answers. 
    It then uses the RAG model to generate answers for the questions.
    It also retrieves the relevant documents for each question.
    Finally, it combines all the information into a dataset object and returns it.

    Args:
        questions: list of strings, questions to be answered
        ground_truth: list of strings, corresponding ground truth answers
        rag_chain: RAG model
        retriever: Retriever object

    Returns:
        dataset: Dataset object containing the questions, answers, contexts and ground truth answers
    '''
    
    data = {"question": [], "answer": [], "contexts": [], "ground_truth": ground_truth}

    for query in questions:
        data["question"].append(query)
        # data["answer"].append(rag_chain.invoke(query)['result'])
        data["answer"].append(rag_chain.invoke(query))
        data["contexts"].append([doc.page_content for doc in retriever.get_relevant_documents(query)])

    dataset = Dataset.from_dict(data)

    return dataset

def evaluate_rag(dataset, llm, embeddings):

    '''
    This function evaluates the RAG model on a dataset using the specified metrics

    Args:
        dataset: Dataset object containing the questions, answers, contexts and ground truth answers
        llm: LLM model
        embeddings: Embeddings object

    Returns:
        result: dictionary containing the evaluation results
    '''
    result = evaluate(
        dataset=dataset,
        llm=llm,
        embeddings=embeddings,
        metrics=[
             
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
        ],
        raise_exceptions=True
    )

    return result