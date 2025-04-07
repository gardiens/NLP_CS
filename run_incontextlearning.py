
import os 
DATA_PATH="/raid/home/bournez_pie/mva_geom/NLP_CS/NLP_CS/data/"
train_data=os.path.join(DATA_PATH,"traindata.csv")
test_data=os.path.join(DATA_PATH,"devdata.csv")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from ollama import chat
from ollama import ChatResponse


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# load the csv  by hand
from src.incontextclassif import * 

ds_train=load_data(train_data)
ds_test=load_data(test_data)
ds_test

ds_train = ds_train.map(remap_aspect)

# Select N examples from training set as demonstrations
demos = ds_train.select([0,1,3])  # example: use 3-shot

# Predict on a single test instance
for i in range(83,84):
    example = ds_test[i]
    pred = predict_sentiment(example, demos,build_prompt=build_prompt)
    #95 à voir
    print("Predicted sentiment:", pred)
    print("True label:", example['polarity'])
    normalize_prediction(pred,example)
example
# from src.clearml import safe_init_clearml
# task=safe_init_clearml(project_name="NLP_CS",task_name="Refined Prompt")

# Predict for all train examples
demonstrations=ds_test
predictions = []
for i in range(len(ds_test)):
    example = ds_test[i]
    print("considered example",example)
    prompt = build_prompt(example, demonstrations)
    
    try:
        response = chat(model='gemma3:1b', messages=[
            {"role": "user", "content": prompt}
        ])
        raw_pred = response['message']['content']
        print("raw pred",raw_pred)
        norm_pred = normalize_prediction(raw_pred,example)
    except Exception as e:
        print("the exception",e)
        norm_pred = "invalid"
    
    predictions.append(norm_pred)
    print(f"[{i+1}/{len(ds_test)}] Predicted: {norm_pred} | True: {example['polarity']}")

# Convert to DataFrame for analysis
results_df = pd.DataFrame({
    "true": [ds_test[i]["polarity"] for i in range(len(ds_test))],
    "predicted": predictions
})

# Accuracy calculation
valid_preds = results_df[results_df["predicted"] != "invalid"]
accuracy = (valid_preds["true"] == valid_preds["predicted"]).mean()
results_df.to_csv("results_incontext.csv", index=False)
print(f"\nIn-Context Learning Accuracy on Train Set (valid predictions only): {accuracy:.4f}")
print(f"Invalid predictions: {len(results_df) - len(valid_preds)} out of {len(results_df)}")
