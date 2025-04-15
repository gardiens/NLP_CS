
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
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
import hydra 
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# load the csv  by hand
from src.incontextclassif import * 

def select_prompt(cfg):
    if cfg.prompt_type=="basic":
        return build_prompt

    else:
        raise ValueError(f"Unknown prompt type: {cfg.prompt_type}")

def get_demonstration(cfg,ds_train,example=None):
    if cfg.demonstration=="basic":
        return ds_train.select([0,1,3])  # example: use 3-shot
    if cfg.demonstration=="None":
        return ds_train.select([])
    if cfg.demonstration=="small":
        return ds_train.select([0,1])  # example: use 3-shot
    
    else:
        raise ValueError(f"Unknown demonstration type: {cfg.demonstration}")
@hydra.main(version_base="1.2", config_path="config", config_name="main.yaml")
def main(cfg ) -> None:

    ds_train=load_data(train_data)
    ds_test=load_data(test_data)
    ds_test

    if cfg.remap_label:
        ds_train = ds_train.map(remap_aspect)
        ds_test= ds_test.map(remap_aspect)
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
    
    from src.clearml import safe_init_clearml,connect_hyperparams_summary
    task=safe_init_clearml(project_name="NLP_CS",task_name=cfg.task_name)
    connect_hyperparams_summary(cfg=cfg,task=task,name="summary config")  # Connect a summary of the config
    demonstrations=get_demonstration(cfg=cfg,ds_train=ds_train,example=example)
    prompt = build_prompt(example, demonstrations)
    print("--- Sample Prompt ---")
    print(prompt)

    # log the sample prompt
    task.get_logger().report_table(
        title="Sample Prompt",
        series="sample prompt",
        table_plot=pd.DataFrame({"prompt": [prompt]}),
    )


    # Predict for all train examples
    # demonstrations=ds_train.select([0,1,3])  # example: use 3-shot
    predictions = []
    maximum=len(ds_test)
    example_info={"polarity":[], "Aspect_Category":[], "Target_term":[],"labels":[],"Sentence":[]}
    build_prompte=select_prompt(cfg)
    model=cfg.model
    print("start training")
    for i in range(maximum):
        example = ds_test[i]
        demonstrations=get_demonstration(cfg=cfg,ds_train=ds_train,example=example)
        prompt = build_prompte(example, demonstrations)
        example_info["polarity"].append(example["polarity"])
        example_info["Aspect_Category"].append(example["Aspect_Category"])
        example_info["Target_term"].append(example["Target_term"])
        example_info["Sentence"].append(example["Sentence"])
        example_info["labels"].append(example["labels"])    
        try:
            # send the message to the chat
            response = chat(model=model, messages=[
                {"role": "user", "content": prompt}
            ])
            raw_pred = response['message']['content']
            print("raw pred",raw_pred)
            norm_pred = normalize_prediction(raw_pred,example)
        except Exception as e:
            print("the exception",e)
            norm_pred = "invalid"
        
        predictions.append(norm_pred)
        print(f"[{i+1}/{maximum}] Predicted: {norm_pred} | True: {example['polarity']}")



    # Convert to DataFrame for analysis
    results_df = pd.DataFrame({
        "true": [ds_test[i]["polarity"] for i in range(maximum)],
        "predicted": predictions
    })

    results_df=pd.concat([pd.DataFrame(example_info),results_df],axis=1)

    # Accuracy calculation
    valid_preds = results_df[results_df["predicted"] != "invalid"]
    accuracy = (valid_preds["true"] == valid_preds["predicted"]).mean()
    results_df.to_csv(cfg.task_name+"results_incontext.csv", index=False)
    print(f"\nIn-Context Learning Accuracy on Train Set (valid predictions only): {accuracy:.4f}")
    print(f"Invalid predictions: {len(results_df) - len(valid_preds)} out of {len(results_df)}")
    task.get_logger().report_scalar("accuracy", "accuracy", value=accuracy,iteration=0)
    task.get_logger().report_scalar("invalid predictions", "invalid predictions", value=len(results_df) - len(valid_preds),iteration=0)
    #! only log the one that are incorrect
    results_df=results_df[results_df["predicted"] != results_df["true"]]

    task.get_logger().report_table(
        title="Results",
        series="results",
        
        table_plot=results_df,
        iteration=0,
    )
if __name__ == "__main__":
    main()