import os
import sys

# Ensure project root is in Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory

from backend.llm_loader import load_falcon_pipeline
from backend.vector_db import create_vector_store


def create_langchain():
  
    # Load Falcon pipeline
    pipeline = load_falcon_pipeline()
    response = pipeline("Tell me about scholarships oppertunities in the Turkiye", max_length=100, do_sample=True, temperature=0.7)
    print(response[0]["generated_text"])
    
    # Wrap as LangChain-compatible LLM
    llm = HuggingFacePipeline(pipeline=pipeline)
    
    # Load prompt template from file
    template_path = os.path.join("templates", "prompt_template.txt")
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    
    prompt = PromptTemplate(template=template, input_variables=["history", "question"])
    
    # Add conversation memory
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    
    # Create the LLMChain
    chain = LLMChain(prompt=prompt, llm=llm, memory=memory)
    
    # Initialize vector store
    vector_store = create_vector_store()
    
    return chain, vector_store


if __name__ == "__main__":
    chain, store = create_langchain()
    print("✅ Falcon pipeline, LLMChain, and vector store initialized successfully!")
