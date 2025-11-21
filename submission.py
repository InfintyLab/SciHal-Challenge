import json
import pandas as pd

def prepare_submission():
    """
    Prepare the submission file based on the predictions from the model.
    """
    task = "subtask1"  
    prediction_file = "output/2025-06-12_pipeline_subtask1_test_llama-3.3-70b-instruct_predictions.json"
    with open(prediction_file, "r") as f:
        predictions = json.load(f)

    if task == "subtask1":
        label_dict = {"entailment": "entail", "contradiction": "contra", "unverifiable": "unver"}
    else:
        label_dict = {
            "entailment": "entail", 
            "unrelated and unverifiable": "unrelunvef", 
            "related but unverifiable": "relunvef",
            "misinterpretation": "misinter", 
            "missing information": "missinfo", 
            "numeric error": "numerr", 
            "negation": "negat", 
            "entity error": "entierr"
        }
            
    id_column = []
    prediction_column = []
    for idx, row in enumerate(predictions):
        id_column.append(row['ID'])
        prediction_column.append(label_dict[row['prediction']])

    # zip the two lists together
    submission_data = pd.DataFrame({
        'ID': id_column,
        'label': prediction_column
    })

    print(submission_data)
    # Save the DataFrame to a CSV file
    submission_file = f"submission/submission_{task}_predictions.csv"
    submission_data.to_csv(submission_file, index=False)

def clean_format():
    """
    Clean the format of the submission file.
    """
    submission_file = "submission/task2_finetunec4_test_predictions_finetune.csv"
    
    # Read the CSV file
    df = pd.read_csv(submission_file,
                 header=None,          # no header row in your file
                 names=["ID", "label"],
                 skipinitialspace=True)  # <-- key line

    print(df.head())
    df.to_csv("submission/0606_task2_finetunec4_test_predictions_finetune.csv", index=False, header=False)


#clean_format()
prepare_submission()