from typing import List
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments
)
from transformers.modeling_outputs import SequenceClassifierOutput

# Disable Weights & Biases logging to save memory and avoid extra logging.
os.environ["WANDB_DISABLED"] = "true"


# ==============================================================================
# 1. Data Loading and Preprocessing Functions
# ==============================================================================

def mark_target_in_sentence(sentence: str, term: str, offsets: str) -> str:
    """
    Inserts special markers [T] and [/T] around the target term using character offsets.
    """
    try:
        start, end = map(int, offsets.split(":"))
        # Insert markers around the term specified by the offsets.
        marked_sentence = sentence[:start] + " [T] " + sentence[start:end] + " [/T] " + sentence[end:]
        return marked_sentence
    except Exception as e:
        # Fall back: Replace all occurrences of the term with marked version.
        return sentence.replace(term, f" [T] {term} [/T] ")

def load_absa_dataset(file_path: str) -> Dataset:
    """
    Reads an ABSA TSV file where each line has 5 tab-separated fields:
      0. Polarity (e.g., positive, negative, neutral)
      1. Aspect_Category (e.g., SERVICE#GENERAL)
      2. Target_term (e.g., wait staff)
      3. Character_offset (e.g., 74:77)
      4. Sentence (the sentence in which the term occurs)

    Returns a Hugging Face Dataset.
    """
    polarity_list = []
    aspect_list = []
    target_list = []
    offset_list = []
    sentence_list = []
    input_texts = []
    polarity_to_label = {"positive": 0, "negative": 1, "neutral": 2}
    labels = []

    skipped = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split("\t")
            if len(tokens) != 5:
                skipped += 1
                continue
            polarity, aspect, target, offset, sentence = tokens
            # Use the character offsets to insert markers around the target term.
            marked_sentence = mark_target_in_sentence(sentence, target, offset)
            input_text = f"Aspect: {aspect}. Sentence: {marked_sentence}"
            polarity_list.append(polarity)
            aspect_list.append(aspect)
            target_list.append(target)
            offset_list.append(offset)
            sentence_list.append(sentence)
            labels.append(polarity_to_label[polarity])
            input_texts.append(input_text)

    print(f"Loaded {len(input_texts)} samples with {skipped} skipped lines from {file_path}.")
    df = pd.DataFrame({
        "polarity": polarity_list,
        "Aspect_Category": aspect_list,
        "Target_term": target_list,
        "Character_offset": offset_list,
        "Sentence": sentence_list,
        "input_text": input_texts,
        "labels": labels
    })
    return Dataset.from_pandas(df)


# ==============================================================================
# 2. Custom RoBERTa Classifier
# ==============================================================================

class CustomRoBERTaClassifier(nn.Module):
    def __init__(self, model_name="roberta-base", num_labels=3):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        # Custom classification head: a two-layer MLP with dropout and ReLU.
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(self.roberta.config.hidden_size, 256)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.out = nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        # Pass through the classification head.
        x = self.dropout(cls_output)
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout2(x)
        logits = self.out(x)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


# ==============================================================================
# 3. Utils Functions (Tokenization, metrics, )
# ==============================================================================

def tokenize_absa(example, tokenizer, max_length=128):
    return tokenizer(example["input_text"], padding="max_length", truncation=True, max_length=max_length)

def compute_metrics(eval_pred):
    """
    Computes accuracy from the logits and labels.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = np.mean(predictions == labels)
    return {"accuracy": accuracy}


# ==============================================================================
# 4. Classifier Wrapper (Train and Predict Methods)
# ==============================================================================

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
        self.model_name = "roberta-base"  # Authorized model, can switch to roberta-large if memory permits.
        print(f"Loading tokenizer for model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print(f"Loading custom classifier model based on: {self.model_name}")
        self.model = CustomRoBERTaClassifier(model_name=self.model_name, num_labels=3)
        self.label_map = {0: "positive", 1: "negative", 2: "neutral"}

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
        print("Loading training dataset...")
        train_dataset = load_absa_dataset(train_filename)
        print("Loading development dataset...")
        dev_dataset = load_absa_dataset(dev_filename)

        print("Tokenizing training data...")
        train_dataset = train_dataset.map(lambda x: tokenize_absa(x, self.tokenizer), batched=False)
        print("Tokenizing development data...")
        dev_dataset = dev_dataset.map(lambda x: tokenize_absa(x, self.tokenizer), batched=False)

        # Set the format for PyTorch
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        dev_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        print(f"Training samples: {len(train_dataset)}")
        print(f"Development samples: {len(dev_dataset)}")

        # Set training arguments with evaluation on dev set at each epoch.
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=10,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=2,
            weight_decay=0.01,
            warmup_steps=150,
            # fp16=True,  # Enable mixed precision training (requires a compatible GPU)
            logging_dir="./logs",
            logging_steps=10,
        )

        print(f"Moving model to device: {device}")
        self.model.to(device)

        # Initialize the Trainer with the compute_metrics function for accuracy.
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=compute_metrics,
        )

        print("Starting training...")
        trainer.train()
        print("Training completed.")

        print("Evaluating on the development set...")
        eval_results = trainer.evaluate()
        print(f"Development Set Accuracy: {eval_results.get('eval_accuracy', 'N/A') * 100:.2f}%")

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
        print("Loading prediction dataset...")
        dataset = load_absa_dataset(data_filename)
        print("Tokenizing prediction data...")
        dataset = dataset.map(lambda x: tokenize_absa(x, self.tokenizer), batched=False)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        print(f"Moving model to device: {device}")
        self.model.to(device)
        self.model.eval()

        predictions = []
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
                predictions.extend(preds)

        predicted_labels = [self.label_map[p] for p in predictions]
        return predicted_labels