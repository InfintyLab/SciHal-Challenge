import os
import re
import json
import requests

from openai import OpenAI
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM


# params
os.environ['HF_TOKEN']=""
#device = torch.device("cuda:1")


class Topic_Aware:
    def __init__(self, client):
        self.client = client

    def query_gpt(self, 
                  message, 
                  model="gpt-4o-2024-08-06", 
                  temperature=0.0,
                  max_tokens=1024
                  ):
        
        response = self.client.chat.completions.create(
                model=model,
                response_format={ "type": "json_object" },
                messages=message,
                max_tokens=max_tokens,
                temperature=temperature
                )
        entity = response.choices[0].message.content
        entity_json = json.loads(entity)
        return entity_json


    def aware(self, text, reference):

        system_prompt= (
            "You are an assistant for topic classification.\n"
            "Given a claim and some reference from academic paper, "
            "please determine which of the following areas the claim belongs to: Computer Science, Engineering, Environmental Science, Medicine, Agricultural and Biological Sciences\n"
            "Return output result in the following JSON format (and nothing else) with the key: topic.\n"
        )       
         
        user_prompt = (
        f"Here is the claim: {text}; reference: {reference} \n"
    )
        
        messages =  [  
                {'role':'system', 'content': system_prompt},    
                {'role':'user', 'content': user_prompt},  
                ]
        return self.query_gpt(messages)
    
    def retry_with_exponential_backoff(self, text, reference, max_retries=5):
        retry_count = 0
        wait_time = 1  # initial wait time in seconds

        while retry_count < max_retries:
            result = self.aware(text, reference)
            if result is not None:
                return result

            time.sleep(wait_time)
            wait_time *= 2  # double the wait time for next retry
            retry_count += 1

        raise Exception("Max retries reached")
    

# zeroshot prompt
# system_prompt= (
#     "You are an assistant for topic classification.\n"
#     "Given a claim and some reference from academic paper, "
#     "please determine which of the following areas the claim belongs to: Computer Science, Engineering, Environmental Science, Medicine, Agricultural and Biological Sciences\n"
#     #"You MUST strictly output your result in the following JSON format (and nothing else).\n"
#     #"with the key: \"prediction\".\n"
#     "Now it's your turn.\n"
# )


if __name__ == "__main__":
    # LLM
    client = OpenAI(api_key="")
    topic_classifier = Topic_Aware(client)
    
    
    # experiment config
    task = "subtask2"  
    batch = "batch2"
    data_type = "test" # if test, no batch
    prompt_setup = "zeroshot" # zeroshot, fewshot, instruct
    
    
    # data loading
    if data_type == "train":
        data_file = f"data/dataset/{task}_{data_type}_{batch}.json"
        data = json.load(open(data_file, "r"))
    else:
        data_file = f"data/dataset/{task}_test.json"
        data = json.load(open(data_file, "r"))


    # running 
    #for idx, row in data.iterrows():
    preds = []
    for idx, row in enumerate(data):
        
        pred = {}
        claim = row['claim']
        reference = row['reference']
        prediction_topic = topic_classifier.retry_with_exponential_backoff(claim, reference)
        print(f"[{idx}] → {prediction_topic}")
        
        if data_type == "train":
            pred['ID'] = row['ID']
            pred['question'] = row['question']
            pred['answer'] = row['answer']
            pred['claim'] = row['claim']
            pred['reference'] = row['reference']
            pred['label'] = row['label']
            pred['justification'] = row['justification']
            pred['topic'] = prediction_topic['topic']
            preds.append(pred)
        
        else:
            pred['ID'] = row['ID']
            pred['question'] = row['question']
            pred['answer'] = row['answer']
            pred['claim'] = row['claim']
            pred['reference'] = row['reference']
            pred['topic'] = prediction_topic['topic']
            preds.append(pred)

    
    #save only the predictions
    output_file = f"data/topic/{task}_{data_type}_{batch}_topic.json"
    with open(output_file, "w") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)

    print(f"Done—wrote {len(preds)} predictions to {output_file}")

