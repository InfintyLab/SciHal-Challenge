import os
import re
import json
import pandas as pd

    

# Convert to JSON format
def convert_to_json(data, task_type, label_dict):
    """
    Convert the given data to JSON format.
    """
    json_data = []
    
    if task_type == "train":
        for index, row in data.iterrows():
            idx = row['ID']
            question = row['question']
            answer = row['answer']
            claim = row['claim']
            #claim_sme = row['claim_sme']
            reference = row['reference']
            label = label_dict[row['label']]
            justification = row['justification']
            json_data.append({"ID": idx,
                            "question": question,
                            "answer": answer,
                            "claim": claim, 
                            #"claim_sme": claim_sme,
                            "reference": reference, 
                            "label": label, 
                            "justification": justification})
    else:
        for index, row in data.iterrows():
            idx = row['ID']
            question = row['question']
            answer = row['answer']
            claim = row['claim']
            reference = row['reference']
            json_data.append({"ID": idx,
                            "question": question,
                            "answer": answer,
                            "claim": claim, 
                            "reference": reference})
        
    return json_data   


# SFT Data: Do classification and justification
def construct_sft_data_cls_just(data, task_type):
    """
    Construct SFT data for the given task.
    """
    sft_data = []

    if task_type == "train":
        for index, row in data.iterrows():
            claim = row['claim']
            reference = row['reference']
            label = label_dict[row['label']]
            justification = row['justification']

            input_text = f"#Claim: {claim}\n #Reference: {reference}"
            output_text = f"#Label: {label}\n #Justification: {justification}"
            instruction = f"Given the claim and reference, classify the claim as unverifiable, contradiction or entailment and provide a justification."
            sft_data.append({"input": input_text, "output": output_text, "instruction": instruction})
    
    else:
        for index, row in data.iterrows():
            claim = row['claim']
            reference = row['reference']
            label = label_dict[row['label']]
            justification = row['justification']

            input_text = f"#Claim: {claim}\n #Reference: {reference}"
            output_text = f"#Label: {label}\n #Justification: {justification}"
            instruction = f"Given the claim and reference, classify the claim as unverifiable, contradiction or entailment and provide a justification."
            sft_data.append({"input": input_text, "output": output_text, "instruction": instruction})
           
    return sft_data


# SFT Data: Just do classification
def construct_sft_data_cls(data, task, task_type):
    """
    Construct SFT data for the given task.
    """
    sft_data = []
    if task == "subtask1":
        if task_type == "train":
            for index, row in data.iterrows():
                claim = row['claim']
                reference = row['reference']
                label = label_dict[row['label']]
                justification = row['justification']

                input_text = f"#Claim: {claim}\n #Reference: {reference}"
                output_text = f"#Label: {label}\n"
                instruction = f"Given the claim and reference, classify the claim as unverifiable, contradiction or entailment."
                sft_data.append({"input": input_text, "output": output_text, "instruction": instruction})
        
        else:
            for index, row in data.iterrows():
                claim = row['claim']
                reference = row['reference']

                input_text = f"#Claim: {claim}\n #Reference: {reference}"
                output_text = f"#Label: "
                instruction = f"Given the claim and reference, classify the claim as unverifiable, contradiction or entailment."
                sft_data.append({"input": input_text, "output": output_text, "instruction": instruction})
    
    else:
        if task_type == "train":
            for index, row in data.iterrows():
                claim = row['claim']
                reference = row['reference']
                label = label_dict[row['label']]
                justification = row['justification']

                input_text = f"#Claim: {claim}\n #Reference: {reference}"
                output_text = f"#Label: {label}\n"
                instruction = f"Given the claim and reference, classify the claim as one of following: entailment, unrelated and unverifiable, misinterpretation, missing information, numeric error, negation, entity error."
                sft_data.append({"input": input_text, "output": output_text, "instruction": instruction})
        
        else:
            for index, row in data.iterrows():
                claim = row['claim']
                reference = row['reference']

                input_text = f"#Claim: {claim}\n #Reference: {reference}"
                output_text = f"#Label: "
                instruction = f"Given the claim and reference, classify the claim as one of following: entailment, unrelated and unverifiable, misinterpretation, missing information, numeric error, negation, entity error."
                sft_data.append({"input": input_text, "output": output_text, "instruction": instruction})

    return sft_data


if __name__ == "__main__":

    # load data
    task = "subtask2"
    batch = "batch2"
    type = "train"  
    if type == "train":
        data_file = f"data/raw_data/{task}_{type}_{batch}.csv"
    else:
        data_file = f"data/raw_data/{task}_test.csv"
    
    if task == "subtask1":
        label_dict = {"entail": "entailment", "contra": "contradiction", "unver": "unverifiable"}
    else:
        label_dict = {
            "entail": "entailment", 
            "unrelunvef": "unrelated and unverifiable", 
            "relunvef": "related but unverifiable",
            "misinter": "misinterpretation", 
            "missinfo": "missing information", 
            "numerr": "numeric error", 
            "negat": "negation", 
            "entierr": "entity error"
        }

    # print raw data sample
    data = pd.read_csv(data_file)
    print(data.head())
    print(f"Loaded {len(data)} questions from {task} dataset")
    print(data.columns)

    # convert to json
    json_data = convert_to_json(data, task_type=type, label_dict=label_dict)

    # save json data
    if type == "train":
        json_data_file = f"data/dataset/{task}_{type}_{batch}.json"
        with open(json_data_file, 'w') as file:
            json.dump(json_data, file, indent=4)
    else:
        json_data_file = f"data/dataset/{task}_test.json"
        with open(json_data_file, 'w') as file:
            json.dump(json_data, file, indent=4)

    # construct sft data
    sft_data = construct_sft_data_cls(data, task, task_type=type)
    # save sft data
    if type == "train":
        sft_data_file = f"data/finetune_data/{task}_{type}_{batch}_cls.json"
        with open(sft_data_file, 'w') as file:
            json.dump(sft_data, file, indent=4)
    else:
        sft_data_file = f"data/finetune_data/{task}_{type}_cls.json"
        with open(sft_data_file, 'w') as file:
            json.dump(sft_data, file, indent=4)


    

