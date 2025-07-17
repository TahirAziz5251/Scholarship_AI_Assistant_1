# import pinecone
# from  langchain.vectorstores import Pinecone
# from langchain.embeddings import HuggingFaceEmbeddings
# from dotenv import load_dotenv
# import os

# load_dotenv()
# def create_vector_store(documents):
#     embeddings = HuggingFaceEmbeddings()
#     pinecone.init(api_key = os.getenv("PINECONE_API_KEY"), environment = os.getenv("PINECONE_ENVIRONMENT"))
#     index = Pinecone.from_texts(documents, embeddings, index_name = "scholarship-ai-assistant-index")
#     return index


import os
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

embeddings = HuggingFaceEmbeddings(model_name="tiiuae/Falcon-H1-3B-Instruct")
api_key = os.getenv("PINECONE_API_KEY")
environment = os.getenv("PINECONE_ENVIRONMENT")  # Make sure this is set in your .env

# Initialize Pinecone client for direct operations
pc = Pinecone(api_key=api_key)

# Connect to a specific index
# Make sure the index "quickstart" already exists in your Pinecone console
index = pc.Index("quickstart")


# Function to create vector store in LangChain from documents
def create_vector_store(documents):
    """
    Create and return a LangChain Pinecone vector store from provided documents.
    """
    # Important: Pinecone must be initialized before using LangChain integration
    # (Here, just make sure your Pinecone project is set up correctly)
    return LangChainPinecone.from_texts(
        documents,
        embeddings,
        index_name="scholarship-ai-assistant-index"  # Use your actual index name
    )


# Direct Pinecone SDK helper to upsert vectors
def upsert_to_index(id, vector):
    """
    Upsert a vector into the Pinecone index.
    """
    index.upsert([(id, vector)])


# Direct Pinecone SDK helper to query vectors
def query_index(vector, top_k=5):
    """
    Query similar vectors from the Pinecone index.
    """
    return index.query(vector=vector, top_k=top_k)

