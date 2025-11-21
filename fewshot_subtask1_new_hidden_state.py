import os
import re
import json
import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

import torch
import pandas as pd
from transformers import pipeline
import numpy as np


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    task: str = "subtask1"
    batch: str = "batch2+batch3"
    data_type: str = "train"
    prompt_setup: str = "fewshot"
    llm: str = "llama-3.1-8b-instruct"
    
    # Generation parameters
    max_new_tokens: int = 64
    temperature: float = 0.6
    top_p: float = 0.9
    do_sample: bool = True


class ModelManager:
    """Manages model loading and inference."""
    
    MODEL_MAPPING = {
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
    
    def __init__(self, model_key: str):
        self.model_key = model_key
        self.model_name = self.MODEL_MAPPING.get(model_key, model_key)
        self.is_instruct = "instruct" in model_key.lower()
        self.generator = None
        
    def load_model(self):
        """Load the model pipeline."""
        try:
            self.generator = pipeline(
                "text-generation",
                model=self.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            if self.generator.tokenizer.pad_token is None:
                self.generator.tokenizer.pad_token = self.generator.tokenizer.eos_token
        except Exception as e:
            raise RuntimeError(f"Error initializing model {self.model_name}: {e}")
    
    def generate(self, prompt: str, config: ExperimentConfig, return_hidden_states: bool = False) -> Dict[str, Any]:
        """Generate text using the loaded model.
        
        Args:
            prompt: Input prompt
            config: Experiment configuration
            return_hidden_states: Whether to return hidden states
            
        Returns:
            Dictionary containing generated text and optionally hidden states
        """
        if self.generator is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        if return_hidden_states:
            # Convert prompt to model inputs
            inputs = self.generator.tokenizer(prompt, return_tensors="pt").to(self.generator.model.device)
            
            # Generate with hidden states
            outputs = self.generator.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.do_sample,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            
            # Extract hidden states from last layer and last token
            hidden_states = outputs.hidden_states[-2][-1].detach().cpu().squeeze()
            
            # Decode generated text
            generated_text = self.generator.tokenizer.batch_decode(
                outputs.sequences, 
                skip_special_tokens=True
            )[0][len(prompt):]
            
            return {
                'generated_text': generated_text,
                'hidden_states': hidden_states
            }
        else:
            outputs = self.generator(
                prompt,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.do_sample,
                return_full_text=False,
                clean_up_tokenization_spaces=True,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            
            return {'generated_text': outputs[0]['generated_text']}
    
    def extract_hidden_states(self, prompt: str, config: ExperimentConfig) -> torch.Tensor:
        """Extract hidden states for a given prompt.
        
        Args:
            prompt: Input prompt
            config: Experiment configuration
            
        Returns:
            Hidden states tensor of shape [hidden_size]
        """
        result = self.generate(prompt, config, return_hidden_states=True)
        return result['hidden_states']


class PromptBuilder:
    """Builds prompts for different setups."""
    
    @staticmethod
    def create_zeroshot_prompt() -> str:
        """Create zero-shot prompt."""
        return (
            "You are an assistant for claim verification.\n"
            "Given a claim and some reference from academic paper, "
            "please classify the claim into three labels: contradiction, entailment, or unverifiable.\n"
            "your answer should be just one word of the three labels.\n"
        )
    
    @staticmethod
    def create_fewshot_prompt(example: str) -> str:
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
    
    @staticmethod
    def create_fewshot_cot_prompt(example: str) -> str:
        """Create few-shot Chain-of-Thought prompt with examples."""
        return (
            "You are an assistant for claim verification.\n"
            "Given a claim and some reference from academic paper, "
            "please classify the claim into three labels: contradiction, entailment, or unverifiable.\n"
            "your answer should be just one word of the three labels.\n"
            "When you classify a claim, please follow these steps:\n"
            "1. Read the reference abstract(s) carefully.\n"
            "2. Read the scientific claim carefully.\n"
            "3. Analyze the relationship between the claim and reference abstract(s).\n"
            "4. Determine which single category best describes the relationship.\n"
            "Here are some examples:\n"
            f"{example}\n\n"
            "Now it's your turn.\n"
        )
    
    @staticmethod
    def get_system_prompt(prompt_type: str, example: str = "") -> str:
        """Get system prompt based on type."""
        if prompt_type == "fewshot":
            return PromptBuilder.create_fewshot_prompt(example)
        elif prompt_type == "cot":
            return PromptBuilder.create_fewshot_cot_prompt(example)
        else:  # zeroshot
            return PromptBuilder.create_zeroshot_prompt()
    
    @staticmethod
    def create_messages(system_prompt: str, claim: str, reference: str) -> List[Dict[str, str]]:
        """Create chat messages for instruct models."""
        query = f"Claim: {claim}\nReference: {reference}\n\nPrediction:"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    
    @staticmethod
    def format_prompt(system_prompt: str, claim: str, reference: str, 
                     is_instruct: bool, tokenizer=None) -> str:
        """Format the complete prompt."""
        if is_instruct and tokenizer:
            messages = PromptBuilder.create_messages(system_prompt, claim, reference)
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            return (f"{system_prompt}"
                   f"Claim: {claim}\n"
                   f"Reference: {reference}\n\n"
                   f"Prediction:")


class DataLoader:
    """Handles data loading operations."""
    
    @staticmethod
    def load_data(config: ExperimentConfig) -> List[Dict[str, Any]]:
        """Load data based on configuration."""
        if config.data_type == "train":
            return DataLoader._load_train_data(config.task)
        else:
            return DataLoader._load_test_data(config.task)
    
    @staticmethod
    def _load_train_data(task: str) -> List[Dict[str, Any]]:
        """Load training data from multiple batches."""
        data = []
        for batch in ["batch2", "batch3"]:
            file_path = f"data/topic/{task}_train_{batch}_topic.json"
            try:
                with open(file_path, "r") as f:
                    data.extend(json.load(f))
            except FileNotFoundError:
                print(f"Warning: Data file {file_path} not found")
        return data
    
    @staticmethod
    def _load_test_data(task: str) -> List[Dict[str, Any]]:
        """Load test data."""
        file_path = f"data/topic/{task}_test_topic.json"
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Data file {file_path} not found")
            return []
    
    @staticmethod
    def load_fewshot_examples() -> Dict[str, str]:
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
                    example_text += (f"Claim: {item['claim']}\n"
                                   f"Reference: {item['reference']}\n"
                                   f"Justification: {item['justification']}\n"
                                   f"Prediction: {item['label']}\n\n")
                
                examples[domain_name] = example_text.strip()
            except FileNotFoundError:
                print(f"Warning: {file_name}.json not found, using empty examples")
                examples[domain_name] = ""
        
        return examples


class OutputManager:
    """Manages output saving and formatting."""
    
    def __init__(self, base_dir: str = "output"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.local_time = datetime.datetime.now().strftime("%Y-%m-%d")
    
    def save_predictions(self, predictions: List[Dict[str, Any]], config: ExperimentConfig) -> str:
        """Save predictions to file."""
        if config.data_type == "train":
            filename = f"{self.local_time}_pipeline_{config.task}_{config.data_type}_{config.batch}_{config.llm}_predictions.json"
        else:
            filename = f"{self.local_time}_pipeline_{config.task}_{config.data_type}_{config.llm}_predictions.json"
        
        output_file = self.base_dir / filename
        
        with open(output_file, "w") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        
        print(f"\nDone—wrote {len(predictions)} predictions to {output_file}")
        return str(output_file)


class ClaimVerificationPipeline:
    """Main pipeline for claim verification."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model_manager = ModelManager(config.llm)
        self.output_manager = OutputManager()
        
        # Set up environment
        os.environ['HF_TOKEN'] = "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"
    
    def run(self):
        """Run the complete pipeline."""
        print(f"Running experiment with config: {self.config}")
        
        # Load data and examples
        data = DataLoader.load_data(self.config)
        fewshot_examples = DataLoader.load_fewshot_examples()
        
        if not data:
            print("No data loaded. Exiting.")
            return
        
        # Load model
        print(f"Loading model: {self.model_manager.model_name}")
        self.model_manager.load_model()
        
        # Run inference
        if self.config.data_type == "train":
            self._run_training(data, fewshot_examples)
        else:
            self._run_testing(data, fewshot_examples)
    
    
    def _run_training(self, data: List[Dict[str, Any]], fewshot_examples: Dict[str, str]):
        """Run training/evaluation with accuracy calculation."""
        predictions = []
        num_correct = 0
        hidden_states_list = []
        labels_list = []
        
        for idx, row in enumerate(data):
            pred = self._process_single_example(idx, row, fewshot_examples)
            predictions.append(pred)
            
            # Store hidden states and labels for potential downstream tasks
            if self.config.data_type == "train" and 'label' in row:
                hidden_states_list.append(pred['hidden_states'])
                labels_list.append(row['label'])
            
            # Calculate accuracy for training data
            if pred['label'] == pred['prediction']:
                num_correct += 1
            
            print(f"[{idx}] Label: {pred['label']}, Prediction: {pred['prediction']}")
        
        # Print results
        total = len(predictions)
        accuracy = num_correct / total if total > 0 else 0
        print(f"\nResults:")
        print(f"Num_Correct: {num_correct}")
        print(f"Num_Total: {total}")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Save predictions and hidden states
        output_file = self.output_manager.save_predictions(predictions, self.config)
        
        # Save hidden states separately if in training mode
        if self.config.data_type == "train" and hidden_states_list:
            hidden_states_file = output_file.replace('_predictions.json', '_hidden_states.npz')
            np.savez(
                hidden_states_file,
                hidden_states=np.array(hidden_states_list),
                labels=np.array(labels_list)
            )
            print(f"Saved hidden states to {hidden_states_file}")
    
    def _run_testing(self, data: List[Dict[str, Any]], fewshot_examples: Dict[str, str]):
        """Run testing without accuracy calculation."""
        predictions = []
        
        for idx, row in enumerate(data):
            pred = self._process_single_example(idx, row, fewshot_examples)
            predictions.append(pred)
            print(f"[{idx}] Prediction: {pred['prediction']}")
        
        self.output_manager.save_predictions(predictions, self.config)
    
    def _process_single_example(self, idx: int, row: Dict[str, Any], 
                               fewshot_examples: Dict[str, str]) -> Dict[str, Any]:
        """Process a single example and return prediction."""
        # Get topic-specific few-shot example
        topic = row.get('topic', '')
        print(f"Processing topic: {topic}")
        example = fewshot_examples.get(topic, "")
        
        # Create system prompt
        system_prompt = PromptBuilder.get_system_prompt(self.config.prompt_setup, example)
        
        # Format prompt
        prompt = PromptBuilder.format_prompt(
            system_prompt, 
            row['claim'], 
            row['reference'],
            self.model_manager.is_instruct,
            self.model_manager.generator.tokenizer if self.model_manager.is_instruct else None
        )
        
        # Generate response with hidden states
        result = self.model_manager.generate(prompt, self.config, return_hidden_states=True)
        raw_output = result['generated_text']
        hidden_states = result['hidden_states']
        
        print("*************")
        print(f"[{idx}] Raw output: {raw_output}")
        
        # Prepare result
        pred = {
            'claim': row['claim'],
            'reference': row['reference'],
            'prediction': raw_output,
            #'hidden_states': hidden_states.numpy()  # Convert to numpy for JSON serialization
        }
        
        # Add additional fields based on data type
        if self.config.data_type == "train" and 'label' in row:
            pred['label'] = row['label']
        elif self.config.data_type == "test" and 'ID' in row:
            pred['ID'] = row['ID']
        
        return pred


def main():
    """Main function to run experiments."""
    # Configure experiment
    config = ExperimentConfig(
        task="subtask1",
        batch="batch3",
        data_type="test",  # "train" or "test"
        prompt_setup="fewshot",  # "zeroshot", "fewshot", "cot"
        llm="llama-3.3-70b-instruct"
    )
    
    # Run pipeline
    pipeline = ClaimVerificationPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
