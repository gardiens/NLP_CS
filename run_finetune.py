import os 
DATA_PATH="NLP_CS/data/"
DATA_PATH="data/"

train_data=os.path.join(DATA_PATH,"traindata.csv")
test_data=os.path.join(DATA_PATH,"devdata.csv")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# from src.clearml import safe_init_clearml
# task=safe_init_clearml(
#     project_name="NLP_CS",
#     task_name="Fine-tune layer",
# )
# load the csv  by hand 
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
    ds_train
    from datasets import Dataset
    ds_train = Dataset.from_pandas(ds_train)
    ds_train
    return ds_train 
ds_train=load_data(train_data)
ds_test=load_data(test_data)
# Reprise du TD de NLP
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel, pipeline
from transformers import logging as hflogging

plm_name="facebook/opt-125m"
# plm_name="google-bert/bert-base-cased"
# Load the config, the tokenizer and the model itself:
lmconfig = AutoConfig.from_pretrained(plm_name)
lmtokenizer = AutoTokenizer.from_pretrained(plm_name)
lm = AutoModel.from_pretrained(plm_name, output_attentions=False)

from transformers import TrainingArguments, Trainer
import numpy as np


class TransformerBinaryClassifier(torch.nn.Module):

    def __init__(self, plm_name: str):
        super(TransformerBinaryClassifier, self).__init__()
        self.lmconfig = AutoConfig.from_pretrained(plm_name)
        self.lmtokenizer = AutoTokenizer.from_pretrained(plm_name)
        self.lm = AutoModel.from_pretrained(plm_name, output_attentions=False)
        self.emb_dim = self.lmconfig.hidden_size
        self.output_size = 1
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(self.emb_dim, self.output_size),
            torch.nn.Sigmoid()
        )
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')


    def forward(self, x):
        x : torch.Tensor = self.lm(x['input_ids'], x['attention_mask']).last_hidden_state
        global_vects = x.mean(dim=1)
        x = self.classifier(global_vects)
        return x.squeeze(-1)

    def compute_loss(self, predictions, target):
        return self.loss_fn(predictions, target)


model = TransformerBinaryClassifier(plm_name)
X_train_encoded = model.lmtokenizer(ds_train["Sentence"],
                            truncation=True,
                            padding=False,
                            add_special_tokens=False,
                            return_tensors=None,
                            return_offsets_mapping=False,
                        )
X_val_encoded = model.lmtokenizer(ds_test["Sentence"],
                            truncation=True,
                            padding=False,
                            add_special_tokens=False,
                            return_tensors=None,
                            return_offsets_mapping=False,
                        )
def tokenize_function(examples):
    return model.lmtokenizer(examples["Sentence"], truncation=True)
def tokenize_function2(examples):
    # Concatenate fields into a single input string
    combined_input = [f"{a} [SEP] {t} [SEP] {s}" for a, t, s in zip(examples["Aspect_Category"], examples["Target_term"], examples["Sentence"])]
    return model.lmtokenizer(combined_input, truncation=True)
def get_tok_ds(ds):
    tok_ds = ds.map(tokenize_function, batched=True)
    tok_ds = tok_ds.remove_columns(["polarity", "Aspect_Category", "Target_term", "Character_offset", "Sentence"])
    return tok_ds
print("start mapping")
tok_ds_train = ds_train.map(tokenize_function2, batched=True)

tok_ds_train = tok_ds_train.remove_columns(["polarity", "Aspect_Category", "Target_term", "Character_offset", "Sentence"])
print("test mapping")
# tok_ds_train = tok_ds_train.rename_column("label", "labels")
tok_ds_test=get_tok_ds(ds_test)
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader

print("we are here")

# just for testing

from torch.optim import AdamW
from transformers import get_scheduler

optimizer = AdamW(model.parameters(), lr=5e-5)
from tqdm.auto import tqdm



def train_model():
    print("start trainingg? 1 ")

    data_collator = DataCollatorWithPadding(tokenizer=model.lmtokenizer, padding=True, return_tensors='pt')

    train_dataloader = DataLoader(tok_ds_train, shuffle=True, batch_size=32, collate_fn=data_collator)# couldn't increase number of workers
    val_dataloader = DataLoader(tok_ds_test, shuffle=False, batch_size=32, collate_fn=data_collator)# couldn't increase number of workers

    num_epochs = 50
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))
    model.train()
    print("start trainingg?")

    for epoch in range(num_epochs):
        correct = 0
        total = 0
        train_loss = 0

        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            predictions = model(batch)
            loss = model.loss_fn(predictions, batch['labels'].float())
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            preds = (predictions > 0.5).float()
            correct += (preds == batch['labels']).sum().item()
            total += batch['labels'].size(0)
            train_loss += loss.item()

        train_accuracy = correct / total
        avg_train_loss = train_loss / len(train_dataloader)

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                predictions = model(batch)
                loss = model.loss_fn(predictions, batch['labels'].float())

                preds = (predictions > 0.5).float()
                val_correct += (preds == batch['labels']).sum().item()
                val_total += batch['labels'].size(0)
                val_loss += loss.item()

        val_accuracy = val_correct / val_total
        avg_val_loss = val_loss / len(val_dataloader)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Acc: {train_accuracy:.4f}, Train Loss: {avg_train_loss:.4f} - Val Acc: {val_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}")

        model.train()
    return model
model= train_model()
# save the model 
save_path = f"saved_model/{plm_name}/"

# Save model weights and config
model.lm.save_pretrained(save_path)

# Save tokenizer
model.lmtokenizer.save_pretrained(save_path)