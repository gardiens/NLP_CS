
import pandas as pd
from datasets import Dataset
import os
from ollama import chat

def load_data(file):
    polarity=[]
    Aspect_Category=[]
    Target_term=[]
    Character_offset=[]
    Sentence=[]
    polarity_to_label={
        "positive":0,
        "negative":1,
        "neutral":2,
    }
    labels=[]
    with open(file) as f:
        for line in f:
            line=line.strip()
            # split by space and remove the \t
            tokens=line.split("\t")
            polarity.append(tokens[0])
            Aspect_Category.append(tokens[1])
            Target_term.append(tokens[2])
            Character_offset.append(tokens[3])
            assert len(tokens[4:])==1,"sentence should be one token,got "+str(len(tokens[4:]))
            Sentence.append(str(tokens[4:][0]))
            labels.append(polarity_to_label[tokens[0]])
    ds_train=pd.DataFrame({"polarity":polarity,
                        "Aspect_Category":Aspect_Category,
                        "Target_term":Target_term,
                        "Character_offset":Character_offset,

                        "labels":labels,
                        "Sentence":Sentence})

    ds_train = Dataset.from_pandas(ds_train)
    return ds_train


aspect_category_mapping = {
    "AMBIENCE#GENERAL": "the general ambience",
    "FOOD#QUALITY": "the quality of the food",
    "SERVICE#GENERAL": "the overall service",
    "FOOD#STYLE_OPTIONS": "the variety or style of food options",
    "DRINKS#QUALITY": "the quality of the drinks",
    "RESTAURANT#MISCELLANEOUS": "miscellaneous aspects of the restaurant",
    "RESTAURANT#GENERAL": "the general experience at the restaurant",
    "DRINKS#PRICES": "the prices of the drinks",
    "FOOD#PRICES": "the prices of the food",
    "LOCATION#GENERAL": "the location of the restaurant",
    "DRINKS#STYLE_OPTIONS": "the variety or style of drink options",
    "RESTAURANT#PRICES": "the prices at the restaurant",
}

def remap_aspect(example, aspect_category_mapping=aspect_category_mapping):
    example["Aspect_Category"] = aspect_category_mapping.get(
        example["Aspect_Category"],
        example["Aspect_Category"]
    )
    return example

def normalize_prediction(pred,example):
    pred = pred.strip().lower()
    valid = ["positive", "negative", "neutral"]
    for label in valid:
        if label in pred:
            return label
    pred2=predict_sentiment(pred,example)
    pred2=pred2.strip().lower()
    if pred2 in valid:
        return pred2
    else:
        return "invalid"


def build_prompt(example, demonstrations):
    prompt = "You are a sentiment analysis model. Classify the sentiment as Positive, Negative, or Neutral.\n"
    for demo in demonstrations:
        prompt += f"Sentence: {demo['Sentence']}\n"
        prompt += f"Aspect: {demo['Aspect_Category']}, Term: {demo['Target_term']}\n"
        prompt += f"Sentiment: {demo['polarity']}\n\n"

    prompt += f"Sentence: {example['Sentence']}\n"
    prompt += f"Aspect: {example['Aspect_Category']}, Term: {example['Target_term']}\n"
    prompt += "Sentiment:"
    return prompt

def predict_sentiment(example, demonstrations,build_prompt=build_prompt):
    prompt = build_prompt(example, demonstrations)
    response = chat(model='gemma3:1b', messages=[
        {"role": "user", "content": prompt}
    ])
    return response['message']['content'].strip()
