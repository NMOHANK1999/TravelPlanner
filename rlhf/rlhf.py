import pandas as pd
import json

from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

import torch
torch.cuda.empty_cache()

combined_df = pd.read_pickle("combined_df.pkl")
def content_role(row):
    prompt = json.loads(row['prompt'].replace('\n', '\\n').replace('\t', '\\t').replace('\\"', '\\\\"'))
    gpt = json.loads(row['gpt'].replace('\n', '\\n').replace('\t', '\\t').replace('\\"', '\\\\"'))
    qwen = row['qwen']
    return {'prompt': prompt[0]['content'], 'chosen': prompt + gpt, 'rejected': prompt + qwen}

train_list = []
for _, row in combined_df.iterrows():
    train_list.append(content_role(row))

train_df = pd.DataFrame(train_list)


model_path = "Qwen/Qwen2-1.5B-Instruct"

#model = AutoModelForCausalLM.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    #attn_implementation="flash_attention_2"
)

tokenizer = AutoTokenizer.from_pretrained(model_path)


# Assuming your data is in a CSV file
train_dataset = Dataset.from_pandas(train_df)



model_output_dir = model_path + str("_DPO")

training_args = DPOConfig(
    output_dir=model_output_dir,
    logging_steps=10,
    per_device_train_batch_size=1,  # Reduce this value
    gradient_accumulation_steps=16,   # Increase this to maintain effective batch size
    fp16=True,
)


from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=4,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

#trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
)
trainer.train()