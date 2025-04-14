from typing import List

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import os
os.environ["WANDB_DISABLED"] = "true"


class Classifier:
    """
    The Classifier: complete the definition of this class template by completing the __init__() function and
    the 2 methods train() and predict() below. Please do not change the signature of these methods
     """


    ############################################# complete the classifier class below
    
    def __init__(self, ollama_url: str):
        """
        This should create and initilize the model.
        !!!!! If the approach you have choosen is in-context-learning with an LLM from Ollama, you should initialize
         the ollama client here using the 'ollama_url' that is provided (please do not use your own ollama
         URL!)
        !!!!! If you have choosen an approach based on training an MLM or a generative LM, then your model should
        be defined and initialized here.
        """
        # self.model_name = "google-bert/bert-base-uncased"
        # self.model_name =  "facebook/opt-125m"
        # self.model_name =  "facebook/opt-350m"
        # self.model_name = "microsoft/deberta-v3-base"
        # self.model_name = "microsoft/deberta-v3-large"
        self.model_name = "FacebookAI/roberta-base"
        # self.model_name = "FacebookAI/roberta-large"
        print(f"Loading tokenizer for model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print(f"Loading model: {self.model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,
            id2label={0: 'positive', 1: 'negative', 2: 'neutral'},
            label2id={'positive': 0, 'negative': 1, 'neutral': 2}
        )
        self.label_map = {'positive': 0, 'negative': 1, 'neutral': 2}

    def _read_data(self, filename: str):
        print(f"Reading data from file: {filename}")
        data = []
        line_count = 0
        skipped_lines = 0
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                line = line.strip()
                if not line:
                    skipped_lines += 1
                    continue
                parts = line.split('\t')
                if len(parts) != 5:
                    skipped_lines += 1
                    continue  # Skip possible malformed lines
                polarity, aspect, term, _, sentence = parts
                input_text = f"Sentence: {sentence}. Term: {term}. Aspect: {aspect}."
                data.append((input_text, polarity))
        print(f"Processed {line_count} lines, skipped {skipped_lines} lines")
        print(f"Loaded {len(data)} valid data samples")
        return data
    
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
        If the approach you have choosen is in-context-learning with an LLM from Ollama, you must
          not train the model, and this method should contain only the "pass" instruction
        Otherwise:
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS

        """
        train_data = self._read_data(train_filename)
        dev_data = self._read_data(dev_filename)
        train_dataset = Dataset.from_dict({
            'text': [item[0] for item in train_data],
            'label': [self.label_map[item[1]] for item in train_data]
        })
        dev_dataset = Dataset.from_dict({
            'text': [item[0] for item in dev_data],
            'label': [self.label_map[item[1]] for item in dev_data]
        })
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(dev_dataset)}")

        tokenized_train = train_dataset.map(
            lambda examples: self.tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128),
            batched=True
        )
        tokenized_dev = dev_dataset.map(
            lambda examples: self.tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128),
            batched=True
        )
        print("Tokenization completed")

        tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        tokenized_dev.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=2,
            # logging_dir='./logs',
            # logging_steps=10,
            weight_decay=0.01,
            warmup_steps=150,
        )
        self.model.to(device)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_dev,
        )
        print("Starting training process...")
        trainer.train()
        print("Training completed successfully")


    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
        If the approach you have choosen is in-context-learning with an LLM from Ollama, ignore the 'device'
        parameter (because the device is specified when launching the Ollama server, and not by the client side)
        Otherwise:
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        print(f"\nStarting prediction")
        print(f"Loading data from: {data_filename}")
        data = self._read_data(data_filename)
        texts = [item[0] for item in data]
        print(f"Processing {len(texts)} samples")

        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        print(f"Input tensors shape: {inputs['input_ids'].shape}")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        self.model.to(device)

        print("Running inference...")
        with torch.no_grad():
            outputs = self.model(**inputs)
        print("Inference completed")
        preds = torch.argmax(outputs.logits, dim=-1)
        predictions = [self.model.config.id2label[pred.item()] for pred in preds]
        return predictions





