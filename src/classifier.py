from typing import List
import torch
import requests
from tqdm import tqdm
import pandas as pd

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
        self.ollama_url = ollama_url
        self.model_name = "gemma3:4b"  
        # self.model_name = "llama3.2:3b"

        # Task instruction
        # self.task_description = (
        #     "You are an aspect-based sentiment analysis system. "
        #     "Your task is to classify the sentiment (positive, negative, or neutral) "
        #     "towards a given aspect term within a sentence. "
        #     "Use the provided character offset to identify the correct term occurrence.\n\n"
        #     "Respond with only one of the following labels: positive, negative, or neutral.\n\n"
        #     "Examples:\n"
        # )

        self.task_description = """
        Perform Aspect-Based Sentiment Analysis. Classify the sentiment as positive, negative, or neutral. Answer in only one word.
        Examples: \n
        """
        # Few-shot examples (could be expanded if needed)
        self.demonstrations = [
            {
                "sentence": "I had fish and my husband had the filet - both of which exceeded our expectations.",
                "term": "filet",
                "aspect": "FOOD#QUALITY",
                "offset": "34:39",
                "label": "positive"
            },
            {
                "sentence": "My quesadilla tasted like it had been made by a three-year old with no sense of proportion or flavor.",
                "term": "quesadilla",
                "aspect": "FOOD#QUALITY",
                "offset": "3:13",
                "label": "negative"
            },
            {
                "sentence": "The food was ok and fair nothing to go crazy.",
                "term": "food",
                "aspect": "FOOD#QUALITY",
                "offset": "4:8",
                "label": "neutral"
            },
            {
                "sentence": "I have never left a restaurant feeling as if I was abused, and wasted my hard earned money.",
                "term": "restaurant",
                "aspect": "RESTAURANT#GENERAL",
                "offset": "20:30",
                "label": "negative"
            }
        ]

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
        pass

    def build_prompt(self, sentence: str, aspect: str, offset: str, category: str, target_term: str) -> str:
        """
        Constructs a prompt for in-context prediction, including few-shot examples.
        """
        prompt = self.task_description

        for ex in self.demonstrations:
            prompt += (
                f"Sentence: {ex['sentence']}\n"
                f"Term: {ex['term']}\n"
                f"Aspect category: {ex['aspect']}\n"
                f"Character offset: {ex['offset']}\n"
                f"Sentiment: {ex['label']}\n\n"
            )
        prompt +="\n Analyze and return the sentiment: \n"
        prompt += (
            f"Sentence: {sentence}\n"
            f"Term: {target_term}\n"
            f"Aspect category: {category}\n"
            f"Character offset: {offset}\n"
            f"Sentiment:"
        )

        return prompt

    def call_ollama(self, prompt: str) -> str:
        """
        Calls Ollama API with a prompt and extracts the predicted sentiment.
        """
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.model_name, "prompt": prompt, "stream": False}
            )

            if response.status_code != 200:
                print("Error:", response.status_code, response.text)
                return "neutral"

            result = response.json().get("response", "").strip().lower()
            if not result in ["positive", "negative", "neutral"]:
                print("Unexpected result:", result)
            if "positive" in result:
                return "positive"
            elif "negative" in result:
                return "negative"
            elif "neutral" in result:
                return "neutral"
            else:
                return "neutral"

        except Exception as e:
            print("Exception during Ollama call:", e)
            return "neutral"

    # def compute_accuracy(self, gold: List[str], predicted: List[str]) -> float:
    #     """
    #     Compares predicted vs true labels.
    #     """
    #     correct = sum(1 for g, p in zip(gold, predicted) if g.strip().lower() == p.strip().lower())
    #     total = len(gold)
    #     return correct / total if total > 0 else 0.0

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
        df = pd.read_csv(data_filename, sep="\t", header=None)
        df.columns = ["polarity", "aspect_category", "target_term", "char_offset", "sentence"]

        y_true = df["polarity"].tolist()
        y_pred = []

        print("Running predictions using Ollama...")

        for _, row in tqdm(df.iterrows(), total=len(df)):
            prompt = self.build_prompt(
                sentence=row["sentence"],
                aspect=row["aspect_category"],
                offset=row["char_offset"],
                category=row["aspect_category"],
                target_term = row["target_term"]
            )
            pred = self.call_ollama(prompt)
            y_pred.append(pred)

        # acc = self.compute_accuracy(y_true, y_pred)
        # print(f"\nAccuracy on the dataset: {acc * 100:.2f}%")
        return y_pred
