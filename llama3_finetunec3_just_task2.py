# may 16
# modified to reduce memory usage 
# may 15
# modified to run on cluster 
# may 15
# actual implementation  
# may 14
# extract hidden states for classification 
# may 6
# modified for task 2
# may 1
# with justification 
# apr 24
# 2nd attempt by using roles 
# apr 24
# modified for few-shot prompting for the scihal dataset
# mar 19
# modified to do TeleQnA
# mar 18
# try out llama3 inference first 

# https://medium.com/@manuelescobar-dev/implementing-and-running-llama-3-with-hugging-faces-transformers-library-40e9754d8c80


import torch
import transformers
import json 
import random 
#from sklearn import metrics
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.linear_model import LogisticRegression
import numpy as np
from transformers import FbgemmFp8Config
import os
from modeling_llama_cluster2 import LlamaForCausalLM

random.seed(42)

# token 
os.environ['HF_TOKEN']=""


#input_filename = "/home/yupcao/data/scihal/subtask1_train_batch2_sft.json"
#input_filename = "/home/yupcao/halsci/dataset/subtask2_train_batch2.json"
input_filename = "/home/datalake/chun-nam/scihal/subtask2_train_batch2.json"
with open(input_filename, "r") as in_file:
    my_json = json.load(in_file)


prompt_template_prefix = """
You are a helpful assistant. Learn from the examples below and complete the task accordingly. 

### Task: Detect if the claims are well-supported by the references. Provide a justification and classify each example into three labels: entailment, contradiction, or neutral

"""

prompt_template_suffix = """

### Now, apply the same pattern: 

Input: !INPUT!
Output: 
"""

#label_dict = {"entail": "entailment", "contra": "contradiction", "neutr": "neutral"}
label_dict = {"entail": "entailment", "negat": "opposite meaning", "relunvef": "related but unverifiable", "misinter": "misinterpretation", "unrelunvef": "unrelated and unverifiable", "entierr": "entity error", "numerr": "numeric error", "missinfo": "missing information"}

label_encoding = dict(zip(label_dict.values(), range(len(label_dict.values()))))

prompt_checklist = [
    "# CHECKLIST", 
    "1. Is the claim related to the references?", 
    "2. Does the claim contain a contradiction to the references?", 
    "3. Does the claim negate parts of the references or replaces terms with their antonyms?",  
    "4. Does the claim present logical fallacies, flawed reasoning (over-claiming, under-claiming, ambiguity, or inconsistency), or illogical conclusions?",  
    "5. Does the claim contain an erroneous numeric value?", 
    "6. Does the claim contain an erroneous entity?",  
    "7. Does the claim omit critical parts from the references, changing the meaning/intent?",  
    "8. Can the claim be supported by the references?" 
]

code_dict = {
             "entailment": ["Yes", "No", "No", "No", "No", "No", "No", "Yes"],
             "opposite meaning": ["Yes", "Yes", "Yes", "No", "No", "No", "No", "No"],
             "related but unverifiable": ["Yes", "No", "No", "No", "No", "No", "No", "No"],
             "misinterpretation": ["Yes", "Yes", "No", "Yes", "No", "No", "No", "Yes"],
             "unrelated and unverifiable": ["No", "NA", "NA", "NA", "NA", "NA", "NA", "NA"],
             "entity error": ["Yes", "Yes", "No", "No", "No", "Yes", "No", "No"],
             "numeric error": ["Yes", "Yes", "No", "No", "Yes", "No", "No", "No"],
             "missing information": ["Yes", "Yes", "No", "No", "No", "No", "Yes", "No"],
             }

def make_checklist(label):
    code = code_dict[label]

    my_checklist = [prompt_checklist[0]] + [x[0]+" "+x[1] for x in zip(prompt_checklist[1:], code)]
    my_checklist = [x+"\n" for x in my_checklist]

    return "".join(my_checklist)


def extract_label(x):
    #my_output = x["output"]
    #s = my_output.find("#Justification")
    #label = my_output[:s]
    #label = label.split()[1].strip()
    #justifiation = my_output[s:]
    #label = x["label"]
    justification = x["justification"]

    return justification


random.shuffle(my_json)
n = int(0.8*len(my_json))
train_dataset = my_json[:n]
eval_dataset = my_json[n:]
#eval_dataset = my_json

#num_entail=0
#num_contra=0
#num_neutr=0
count_dict = {}
for label in label_dict:
    count_dict[label_dict[label]]=0


prompt_middle = ""
fs_total = 0
for x in train_dataset:
    label = x["label"]
    justification = extract_label(x)
    claim = x["claim"]
    reference = x["reference"]
    my_input = "#Claim: "+claim+"\n #Reference: "+reference
    if count_dict[label]<2:
        #prompt_middle += "Input: "+my_input+"\n\n"+"Output:\n"+justification+"\n#Label: "+label+"\n\n"
        prompt_middle += "Input: "+my_input+"\n\n"+"Output:\n"+justification+"\n\n"+make_checklist(label)+"\n\n#Label: "+label+"\n\n"
        count_dict[label]+=1
        fs_total+=1
    if fs_total==16:
        break


#prompt_template = prompt_template_prefix + prompt_middle + prompt_template_suffix

#model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"


#quantization_config = BitsAndBytesConfig(load_in_8bit=True)
quantization_config = FbgemmFp8Config()
#model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0", quantization_config=quantization_config, torch_dtype=torch.bfloat16)
model = LlamaForCausalLM.from_pretrained(model_id, device_map="cuda:0", quantization_config=quantization_config, torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")


#pipeline = transformers.pipeline(
#    "text-generation",
#    model=model_id,
#    model_kwargs={
#        "torch_dtype": torch.float16,
#        "quantization_config": {"load_in_8bit": True},
#        "low_cpu_mem_usage": True,
#    },
#)


#terminators = [
#    pipeline.tokenizer.eos_token_id,
    #pipeline.tokenizer.convert_tokens_to_ids(""),
#]


#def get_response(query, message_history=[], max_tokens=8192, temperature=0.6, top_p=0.9):
def get_response(query, message_history=[], max_tokens=16384, temperature=0.6, top_p=0.9):
    user_prompt = message_history + [{"role": "system", "content": prompt_template_prefix}] + [{"role": "user", "content": query}]
    #user_prompt += [{"role": "system", "content": prompt_template_prefix}]
    prompt = pipeline.tokenizer.apply_chat_template(
        user_prompt, tokenize=False, add_generation_prompt=True
    )
    #print(prompt)
    #print("\n\n\n\n\n")
    #prompt = prompt_template.replace("!INPUT!", query)
    outputs = pipeline(
        prompt,
        max_new_tokens=max_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )
    response = outputs[0]["generated_text"][len(prompt):]
    return response, user_prompt + [{"role": "assistant", "content": response}]
    #return response


#my_response, _ = get_response("What is the capital of France?")

#question_list = list(my_json.keys())
#question_list = question_list[:10]

def find_label(x):
    s = x.find("#Label")
    if s!=-1:
        x = x[s:]

    if x.find("entailment")!=-1:
        return "entailment"
    elif x.find("opposite")!=-1:
        return "opposite meaning"
    elif x.find("related but unverifiable")!=-1:
        return "related but unverifiable"
    elif x.find("unrelated and unverifiable")!=-1:
        return "unrelated and unverifiable"
    elif x.find("misinterpretation")!=-1:
        return "misinterpretation"
    elif x.find("entity")!=-1 and x.find("error")!=-1:
        return "entity error"
    elif x.find("numeric")!=-1 and x.find("error")!=-1:
        return "numeric error"
    elif x.find("missing")!=-1 and x.find("information")!=-1:
        return "missing information"
    else:
        return "unknown"

num_correct=0
k=0
count_dict = {}
correct_dict = {}
predict_dict = {}
for label in label_dict:
    count_dict[label_dict[label]]=0
    correct_dict[label_dict[label]]=0
    predict_dict[label_dict[label]]=0
predict_dict["unknown"]=0

embedding_list = []
label_list = []
for q in train_dataset:
    #my_input = q["input"]
    #print(full_question)
    label, justification = extract_label(q)
    claim = q["claim"]
    reference = q["reference"]
    my_input = "#Claim: "+claim+"\n #Reference: "+reference

    my_prompt_middle = prompt_middle + prompt_template_suffix.replace("!INPUT!", my_input)

    user_prompt = [{"role": "system", "content": prompt_template_prefix}] + [{"role": "user", "content": my_prompt_middle}]

    prompt = tokenizer.apply_chat_template(user_prompt, tokenize=False, add_generation_prompt=True)
    prompt_length = len(prompt)
    prompt = tokenizer(prompt, return_tensors="pt").to("cuda")

    generated_ids = model.generate(**prompt, max_new_tokens=1024, temperature=0.6, top_p=0.9, do_sample=True, eos_token_id=[tokenizer.eos_token_id], output_hidden_states=True, return_dict_in_generate=True)
    #print(generated_ids)
    hidden_states = generated_ids.hidden_states
    generated_ids = generated_ids.sequences

    my_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    my_response = my_output[prompt_length:]
    print(my_response)

    embedding = hidden_states[-2][-1].detach().cpu().squeeze()
    embedding_list.append(embedding)
    label_list.append(label_encoding[label])

    pred = find_label(my_response)

    if pred==label:
        num_correct+=1
        correct_dict[label]+=1

    count_dict[label]+=1
    predict_dict[pred]+=1


    k+=1

    if k>10:
        break 


lr_features = torch.stack(embedding_list).float().numpy()
print(lr_features.size())
lr_labels = torch.tensor(label_list).numpy()

logreg = LogisticRegression(C=1)
logreg.fit(lr_features,lr_labels)


k=0
eval_embedding_list = []
eval_label_list = []
# evaluation data 
for q in eval_dataset:
    #my_input = q["input"]
    #print(full_question)
    label, justification = extract_label(q)
    claim = q["claim"]
    reference = q["reference"]
    my_input = "#Claim: "+claim+"\n #Reference: "+reference

    my_prompt_middle = prompt_middle + prompt_template_suffix.replace("!INPUT!", my_input)

    user_prompt = [{"role": "system", "content": prompt_template_prefix}] + [{"role": "user", "content": my_prompt_middle}]

    prompt = tokenizer.apply_chat_template(user_prompt, tokenize=False, add_generation_prompt=True)
    prompt_length = len(prompt)
    prompt = tokenizer(prompt, return_tensors="pt").to("cuda")

    generated_ids = model.generate(**prompt, max_new_tokens=1024, temperature=0.6, top_p=0.9, do_sample=True, eos_token_id=[tokenizer.eos_token_id], output_hidden_states=True, return_dict_in_generate=True)
    #print(generated_ids)
    hidden_states = generated_ids.hidden_states
    generated_ids = generated_ids.sequences

    my_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    my_response = my_output[prompt_length:]
    print(my_response)

    embedding = hidden_states[-2][-1].detach().cpu().squeeze()
    eval_embedding_list.append(embedding)
    eval_label_list.append(label_encoding[label])

    k+=1

    if k>10:
        break 



eval_lr_features = torch.stack(eval_embedding_list).float().numpy()
print(eval_lr_features.size())
eval_lr_labels = torch.tensor(eval_label_list).numpy()

predictions = logreg.predict(eval_lr_features)


output_filename = "/home/datalake/chun-nam/scihal/lr_features.npz"
np.savez(output_filename, lr_features, lr_labels, eval_lr_features, eval_lr_labels)



print(num_correct)
print(k)

print(correct_dict)
print(count_dict)
print(predict_dict)




