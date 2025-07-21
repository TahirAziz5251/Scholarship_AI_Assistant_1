import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone


api_key = os.getenv("PINECONE_API_KEY")
index_name = "scholarship-ai-assistant-index" 
embedding_model_name = "thenlper/gte-large"   
embedding_dimension = 1024
metric = "cosine"
cloud = "aws"
region = "us-east-1"



# Initialize Pinecone client
pc = Pinecone(api_key=api_key, environment=os.getenv("PINECONE_ENV"))

existing_indexes = [idx.name for idx in pc.list_indexes()]

if index_name not in existing_indexes:
    print("[INFO] Creating index...")
    pc.create_index(
        name=index_name,
        spec={
            "serverless": {"cloud": cloud, "region": region},
            "dimension": embedding_dimension,
            "metric": metric
        }
    )
    print(f"[INFO] Created index: {index_name}")
else:
    print(f"[INFO] Index already exists: {index_name}")

index = pc.Index(index_name)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

def create_vector_store(documents):
    return LangChainPinecone.from_texts(
        documents,
        embeddings,
        index_name=index_name
    )


def upsert_to_index(id, vector):
    """
    Upsert a single vector into the Pinecone index.
    
    Args:
        id (str): Unique ID for the vector.
        vector (list or np.ndarray): Embedding vector.
    """
    index.upsert([(id, vector)])
    print(f"[INFO] Upserted vector with ID: {id}")


def query_index(vector, top_k=5):
    """
    Query similar vectors from the Pinecone index.
    
    Args:
        vector (list or np.ndarray): Query embedding.
        top_k (int): Number of top results to retrieve.
        
    Returns:
        Query results from Pinecone.
    """
    return index.query(vector=vector, top_k=top_k)

