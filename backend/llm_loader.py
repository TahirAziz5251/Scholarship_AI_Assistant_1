from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
def load_falcon_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/Falcon-H1-3B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("tiiuae/Falcon-H1-3B-Instruct", torch_dtype=torch.float16, device_map="auto")
    device = 0 if torch.cuda.is_available() else -1
    text_gen_pipeline = pipeline("text-generation", model= model, tokenizer=tokenizer, device=device)

    return text_gen_pipeline