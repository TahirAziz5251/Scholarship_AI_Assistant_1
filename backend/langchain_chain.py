from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline  
from backend.llm_loader import load_falcon_pipeline
from langchain.memory import ConversationBufferMemory

def create_langchain():
    pipeline = load_falcon_pipeline()
    llm = HuggingFacePipeline(pipeline=pipeline)
    with open("templates/prompt_template.txt", 'r') as f:
        template = f.read()
        prompt = PromptTemplate(template=template, input_variables=["history", "question"])

        # Now add conversation memory
        memory = ConversationBufferMemory(memory_key="history", return_messages=True)
            
        chain = LLMChain(prompt=prompt, llm=llm)
        return chain
