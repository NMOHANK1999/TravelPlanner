import argparse
import random
import numpy as np
from datasets import load_dataset, Dataset
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer
#import wandb
import pandas as pd
import gc
import os
from datetime import datetime
import json
import ipdb
import wandb




gc.collect()
torch.cuda.empty_cache()


def seed_everything(seed=2003):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def apply_lora(model):
    lora_config = LoraConfig(
        r=4,
        lora_alpha=32,
        lora_dropout=0.05,
        #target_modules="all-linear"
        target_modules = 'all-linear'
    )
    model = get_peft_model(model, lora_config)
    return model


def preprocess_data(item):
    return {
        'prompt': 'Instruct: ' + item['prompt'] + '\n',
        'chosen': 'Output: ' + item['chosen'],
        'rejected': 'Output: ' + item['rejected']
    }


def main():

    model_name = "hsaest/Llama-3.1-8B-Instruct-travelplanner-SFT"

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lambda_penalty", type=float, default=0.1)
    parser.add_argument("--step_size", type=float, default=0.01)
    parser.add_argument("--threshold", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=2003)


    #parser.add_argument("--model_name", type=str, default="gpt2-medium") #gpt-2 with truthy-dpo
    #parser.add_argument("--dataset_name", type=str, default="Intel/orca_dpo_pairs")     # jondurbin/truthy-dpo-v0.1
    parser.add_argument("--wandb_project", type=str, default="DPO_GEN_AI_assignment")


    args = parser.parse_args()


    print(args)
    seed_everything(args.seed)

    wandb.login()  
    wandb_run_name = f"constr-dpo:-bs:{args.batch_size}-epochs:{args.epochs}"
    wandb.init(project=args.wandb_project, name=wandb_run_name, config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)


    ref_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)


    model = apply_lora(base_model)

    # Data Loading and preprecessing for Intel/orca_dpo_pairs (subset)
    # train_dataset = load_dataset(args.dataset_name, split="train[:10%]")
    # val_dataset = load_dataset(args.dataset_name, split="train[10%:13%]")
    # train_dataset = train_dataset.map(preprocess_data)
    # val_dataset = val_dataset.map(preprocess_data)    

    # Load bad dataset (words) -> Tokenize bad words
    #bad_token_ids = load_bad_words(tokenizer)


    combined_df = pd.read_pickle("data/combined_df_full.pkl")
    
    
    def content_role(row):
        print()
        prompt = json.loads(row['prompt'].replace('\n', '\\n').replace('\t', '\\t').replace('\\"', '\\\\"'))
        gpt = json.loads(row['gpt'].replace('\n', '\\n').replace('\t', '\\t').replace('\\"', '\\\\"'))
        qwen = row['qwen']        
        return {'prompt': prompt[0]['content'], 'chosen': gpt[0]['content'], 'rejected': qwen[0]['content']}

    train_list = []
    for _, row in combined_df.iterrows():
        train_list.append(content_role(row))

    train_df = pd.DataFrame(train_list)
    
    train_dataset = Dataset.from_pandas(train_df)

    train_dataset = train_dataset.map(preprocess_data)


    result_name = str(model_name) + "_results"
    output_dir = os.path.abspath(result_name)


    training_args = TrainingArguments(
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        report_to="wandb",
        output_dir=output_dir,
        overwrite_output_dir=True,
        logging_steps=10,
        remove_unused_columns=False,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        logging_dir='./logs',
        #extra_shit
        logging_strategy="steps",
        save_strategy="steps",
        save_steps=50,
    )

    model.train()


    ref_model.eval()


    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        #beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        tokenizer=tokenizer,
        args=training_args,
        #bad_token_ids=bad_token_ids,
        #lambda_val=args.lambda_penalty, 
        #step_size=args.step_size,
        #threshold=args.threshold,  
        #max_length=1024,
        #max_prompt_length=512
    )

    trainer.train()


    #timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")


    model_save_path = f"DPO_{model_name}.pt"
    
    
    model.save_pretrained(model_save_path)

if __name__ == "__main__":
    main()