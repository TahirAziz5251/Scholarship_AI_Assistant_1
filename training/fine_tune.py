from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import json
import torch
from torch.utils.data import Dataset

# Load dataset
with open('../data/scholarship_qa_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Define custom Dataset
class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, padding='max_length'):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instruction = self.data[idx]['instruction']
        response = self.data[idx]['response']
        full_text = f"Instruction: {instruction}\nResponse: {response}"

        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids
        }

# Load tokenizer & model
model_name = "tiiuae/Falcon-H1-3B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Apply PEFT with LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Create Dataset & DataCollator
dataset = QADataset(data, tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="../models/fine_tuned_model",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    logging_dir="../logs",
    logging_steps=10,
    save_steps=500,
    learning_rate=5e-5,
    fp16=True
)

#  Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

#  Start fine-tuning
trainer.train()

#  Save the fine-tuned model
model.save_pretrained("../models/fine_tuned_model")
tokenizer.save_pretrained("../models/fine_tuned_model")

