import os
import re
import json
import datetime

import torch
import pandas as pd
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# params
local_time = datetime.datetime.now().strftime("%Y-%m-%d")
# Load HF_TOKEN from environment variables (from .env file)
if 'HF_TOKEN' not in os.environ:
    raise ValueError("HF_TOKEN not found in environment variables. Please check your .env file.")


def parse_model_output(text: str):
    """Parse model output to extract prediction."""
    # Look for "Prediction:" at the start of a line
    for line in text.splitlines():
        if line.startswith("Prediction:"):
            value = line.split("Prediction:", 1)[1].strip()
            return {"prediction": value}

    # Fallback: regex search
    match = re.search(r'Prediction:\s*([^\n]+)', text)
    return {"prediction": match.group(1).strip()} if match else {"prediction": ""}


def load_fewshot_examples():
    """Load and construct few-shot examples for each domain."""
    examples = {}
    domains = {
        "Computer Science": "computer_science",
        "Engineering": "engineering", 
        "Agricultural and Biological Sciences": "biological",
        "Environmental Science": "environmental",
        "Medicine": "medicine"
    }
    
    for domain_name, file_name in domains.items():
        try:
            with open(f"fewshot_example/subtask1/{file_name}.json", "r") as f:
                data = json.load(f)
            
            example_text = ""
            for item in data:
                example_text += f"Claim: {item['claim']}\n"
                example_text += f"Reference: {item['reference']}\n"
                example_text += f"Justification: {item['justification']}\n"
                example_text += f"Prediction: {item['label']}\n\n"
            
            examples[domain_name] = example_text.strip()
        except FileNotFoundError:
            print(f"Warning: {file_name}.json not found, using empty examples")
            examples[domain_name] = ""
    
    return examples


def create_fewshot_prompt_task1(example):
    """Create few-shot prompt with examples."""
    return (
        "You are an assistant for claim verification.\n"
        "Given a claim and some reference from academic paper, "
        "please classify the claim into three labels: contradiction, entailment, or unverifiable.\n"
        "your answer should be just one word of the three labels.\n"
        "Here are some examples:\n"
        f"{example}\n\n"
        "Now it's your turn.\n"
    )
    
    
def create_fewshot_cot_prompt_task1(example):
    """Create few-shot prompt with examples."""
    return (
        "You are an assistant for claim verification.\n"
        "Given a claim and some reference from academic paper, "
        "please classify the claim into three labels: contradiction, entailment, or unverifiable.\n"
        "your answer should be just one word of the three labels.\n"
        "please think step by step.\n"
        "Here are some examples:\n"
        f"{example}\n\n"
        "Now it's your turn.\n"
    )


def create_chat_messages(system_prompt, claim, reference):
    """Create chat messages for instruct models."""
    query = f"Claim: {claim}\nReference: {reference}\n\nPrediction:"
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]


def get_model_name(llm_key):
    model_mapping = {
        "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
        "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
        "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
        "llama-3.1-70b": "meta-llama/Llama-3.1-70B",
        "llama-3-70b-instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
        "llama-3.1-70b-instruct": "meta-llama/Llama-3.1-70B-Instruct",
        "llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
        "qwen-2.5-7b": "Qwen/Qwen2.5-7B",
        "deepseek-r1-llama-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    }
    return model_mapping.get(llm_key, llm_key)


def train(task, 
        batch, 
        data_type, 
        prompt_setup, 
        data, 
        MODEL_NAME, 
        fewshot_examples):
        
    # Initialize text generation pipeline
    try:
        generator = pipeline(
            "text-generation",
            model=MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,  # Some models may need this
        )
        if generator.tokenizer.pad_token is None:
            generator.tokenizer.pad_token = generator.tokenizer.eos_token
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    # Check if it's an instruct model
    is_instruct_model = "instruct" in LLM.lower()
    
    # Running inference
    preds = []
    num_correct = 0
    for idx, row in enumerate(data):
        pred = {}
        
        # Get topic-specific few-shot example
        topic = row.get('topic')
        print(f"Processing topic: {topic}")
        example = fewshot_examples.get(topic)
        
        # Create system prompt
        system_prompt = create_fewshot_prompt_task1(example)
        
        if is_instruct_model:
            # Use chat template for instruct models
            messages = create_chat_messages(system_prompt, row['claim'], row['reference'])
            prompt = generator.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Simple concatenation for base models
            prompt = system_prompt
            prompt += f"Claim: {row['claim']}\n"
            prompt += f"Reference: {row['reference']}\n"
            prompt += "\nPrediction:"
        
        # Generate response
        outputs = generator(
            prompt,
            max_new_tokens=64,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            return_full_text=False,
            clean_up_tokenization_spaces=True,
            pad_token_id=generator.tokenizer.eos_token_id
        )
        
        # Extract and parse output
        print("*************")
        raw_output = outputs[0]['generated_text']
        print(f"[{idx}] Raw output: {raw_output}")     
        print(f"[{idx}] Prediction: {raw_output}")
        
        # Store results
        pred['claim'] = row['claim']
        pred['reference'] = row['reference']
        pred['label'] = row['label']
        pred['prediction'] = raw_output
        preds.append(pred)

        if pred['label'] == pred['prediction']:
            num_correct += 1
        
        print(f"[{idx}] Processed label: {pred['label']}")
        print(f"[{idx}] Processed pred: {pred['prediction']}")
                
            
    total = len(preds)
    accuracy = num_correct / total if total > 0 else 0
    print(f"\nResults:")
    print(f"Num_Correct: {num_correct}")
    print(f"Num_Total: {total}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Save predictions
    os.makedirs("output", exist_ok=True)
    output_file = f"output/{local_time}_pipeline_{task}_{data_type}_{batch}_{LLM}_predictions.json"
    
    with open(output_file, "w") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)
    print(f"\nDone—wrote {len(preds)} predictions to {output_file}")


def test(task, 
        batch, 
        data_type, 
        prompt_setup, 
        data, 
        MODEL_NAME, 
        fewshot_examples):
    
    # Initialize text generation pipeline
    try:
        generator = pipeline(
            "text-generation",
            model=MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,  # Some models may need this
        )
        if generator.tokenizer.pad_token is None:
            generator.tokenizer.pad_token = generator.tokenizer.eos_token
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    # Check if it's an instruct model
    is_instruct_model = "instruct" in LLM.lower()
    
    # Running inference
    preds = []
    num_correct = 0
    
    for idx, row in enumerate(data):
        pred = {}
        
        # Get topic-specific few-shot example
        topic = row.get('topic')
        print(f"Processing topic: {topic}")
        example = fewshot_examples.get(topic)
        
        # Create system prompt
        system_prompt = create_fewshot_prompt_task1(example)
        
        if is_instruct_model:
            # Use chat template for instruct models
            messages = create_chat_messages(system_prompt, row['claim'], row['reference'])
            prompt = generator.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Simple concatenation for base models
            prompt = system_prompt
            prompt += f"Claim: {row['claim']}\n"
            prompt += f"Reference: {row['reference']}\n"
            prompt += "\nPrediction:"
        
        # Generate response
        outputs = generator(
            prompt,
            max_new_tokens=64,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            return_full_text=False,
            clean_up_tokenization_spaces=True,
            pad_token_id=generator.tokenizer.eos_token_id
        )
        
        # Extract and parse output
        print("*************")
        raw_output = outputs[0]['generated_text']
        print(f"[{idx}] Raw output: {raw_output}")
        print(f"[{idx}] Prediction: {raw_output}")
        
        # Store results
        pred['ID'] = row['ID']
        pred['claim'] = row['claim']
        pred['reference'] = row['reference']
        pred['prediction'] = raw_output
        preds.append(pred)
        print(f"[{idx}] Processed pred: {pred['prediction']}")
    
    # Save predictions
    os.makedirs("output", exist_ok=True)
    output_file = f"output/{local_time}_pipeline_{task}_{data_type}_{LLM}_predictions.json"
    
    with open(output_file, "w") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)
    print(f"\nDone—wrote {len(preds)} predictions to {output_file}")
    
        
if __name__ == "__main__":
    
    # Experiment config
    task = "subtask1"  
    batch = "batch2+batch3"  # batch2, batch3, batch2+batch3
    data_type = "train"  # if test, no batch
    prompt_setup = "fewshot"  # zeroshot, fewshot, cot
    LLM = "llama-3.1-8b-instruct"
    
    # Data loading
    if data_type == "train":
        data_file1 = f"data/topic/{task}_{data_type}_batch2_topic.json"  
        data_file2 = f"data/topic/{task}_{data_type}_batch3_topic.json"
        try:
            with open(data_file1, "r") as f:
                data1 = json.load(f)
        except FileNotFoundError:
            print(f"Error: Data file {data_file} not found")
        try:
            with open(data_file2, "r") as f:
                data2 = json.load(f)
        except FileNotFoundError:
            print(f"Error: Data file {data_file} not found")
        data = data1 + data2
    else:
        data_file = f"data/topic/{task}_test_topic.json"
        try:
            with open(data_file, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Data file {data_file} not found")

    
    # Load few-shot examples
    fewshot_examples = load_fewshot_examples()
    
    # # Get model name
    MODEL_NAME = get_model_name(LLM)
    print(f"Using model: {MODEL_NAME}")
    
    if data_type == "train":
        train(task, batch, data_type, prompt_setup, data, MODEL_NAME, fewshot_examples)
    else:
        test(task, batch, data_type, prompt_setup, data, MODEL_NAME, fewshot_examples)
        