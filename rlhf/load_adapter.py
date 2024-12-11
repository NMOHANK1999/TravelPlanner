from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2-1.5B-Instruct"
peft_model_id = "DPO_Qwen/Qwen2-1.5B-Instruct.pt"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.load_adapter(peft_model_id)

save_directory = f"rlhf_adapter/{model_id}"

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

model.config.save_pretrained(save_directory)

print(f"Base model saved to {save_directory}")