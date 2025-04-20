from typing import List
import torch
import requests
import pandas as pd
from tqdm import tqdm

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
        self.prompts = [self.prompt_style_1, self.prompt_style_2, self.prompt_style_3]

        self.prompt_instruction = """
        Perform Aspect-Based Sentiment Analysis. Classify the sentiment as positive, negative, or neutral. Answer in only one word.
        Examples: \n
        """

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
        pass  # Not used in in-context learning

    def call_ollama(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.model_name, "prompt": prompt,"temperature":0, "stream": False}
            )
            if response.status_code != 200:
                print("Error:", response.status_code, response.text)
                return "neutral"

            result = response.json().get("response", "").strip().lower()
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

    def majority_vote(self, predictions: List[str]) -> str:
      count_dict = {}
      for pred in predictions:
          if pred in count_dict:
              count_dict[pred] += 1
          else:
              count_dict[pred] = 1

      max_count = -1
      majority_label = None
      for label, count in count_dict.items():
          if count > max_count:
              max_count = count
              majority_label = label

      return majority_label


    def prompt_style_1(self, sentence, target_term, aspect, offset):
        examples =  [
            # ("My quesadilla tasted like it had been made by a three-year old with no sense of proportion or flavor.", "quesadilla", "FOOD#QUALITY", "3:13", "negative"),
            # ("This place is incredibly tiny.", "place", "RESTAURANT#MISCELLANEOUS", "5:10", "negative"),
            # ("I found the food to be outstanding, particulary the salmon dish I had.", "food", "FOOD#QUALITY", "12:16", "positive"),
            ("Service was prompt and courteous.", "Service", "SERVICE#GENERAL", "0:7", "positive"),
            # # ("delicious bagels, especially when right out of the oven.", "bagels", "FOOD#QUALITY", "10:16", "positive"),
            # ("I have never left a restaurant feeling as if i was abused, and wasted my hard earned money.", "restaurant", "RESTAURANT#GENERAL", "20:30", "negative"),
            # ("The food was ok and fair nothing to go crazy.", "food", "FOOD#QUALITY", "4:8", "neutral"),
            # ("overpriced japanese food with mediocre service", "service", "SERVICE#GENERAL", "39:46", "neutral")
        ]

        prompt = self.prompt_instruction
        for ex in examples:
            prompt += (
                f"Sentence: {ex[0]}\n"
                f"Term: {ex[1]}\n"
                f"Aspect category: {ex[2]}\n"
                f"Character offset: {ex[3]}\n"
                f"Sentiment: {ex[4]}\n\n"
            )
        prompt +="\n Analyze and return the sentiment: \n"
        prompt += (
            f"Sentence: {sentence}\n"
            f"Term: {target_term}\n"
            f"Aspect category: {aspect}\n"
            f"Character offset: {offset}\n"
            f"Sentiment:"
        )
        return prompt

    def prompt_style_2(self, sentence, target_term, aspect, offset):
        examples =  [
            # ("My quesadilla tasted like it had been made by a three-year old with no sense of proportion or flavor.", "quesadilla", "FOOD#QUALITY", "3:13", "negative"),
            # ("This place is incredibly tiny.", "place", "RESTAURANT#MISCELLANEOUS", "5:10", "negative"),
            # ("I found the food to be outstanding, particulary the salmon dish I had.", "food", "FOOD#QUALITY", "12:16", "positive"),
            # ("Service was prompt and courteous.", "Service", "SERVICE#GENERAL", "0:7", "positive"),
            # ("delicious bagels, especially when right out of the oven.", "bagels", "FOOD#QUALITY", "10:16", "positive"),
            ("I have never left a restaurant feeling as if i was abused, and wasted my hard earned money.", "restaurant", "RESTAURANT#GENERAL", "20:30", "negative"),
            # ("The food was ok and fair nothing to go crazy.", "food", "FOOD#QUALITY", "4:8", "neutral"),
            # ("overpriced japanese food with mediocre service", "service", "SERVICE#GENERAL", "39:46", "neutral")
        ]

        prompt = self.prompt_instruction
        for ex in examples:
            prompt += (
                f"Sentence: {ex[0]}\n"
                f"Term: {ex[1]}\n"
                f"Aspect category: {ex[2]}\n"
                f"Character offset: {ex[3]}\n"
                f"Sentiment: {ex[4]}\n\n"
            )
        prompt +="\n Analyze and return the sentiment: \n"
        prompt += (
            f"Sentence: {sentence}\n"
            f"Term: {target_term}\n"
            f"Aspect category: {aspect}\n"
            f"Character offset: {offset}\n"
            f"Sentiment:"
        )
        return prompt

    def prompt_style_3(self, sentence, target_term, aspect, offset):
        examples =  [
            # ("I had fish and my husband had the filet - both of which exceeded our expectations.","filet","FOOD#QUALITY","34:39","positive"),
            # ("My quesadilla tasted like it had been made by a three-year old with no sense of proportion or flavor.", "quesadilla", "FOOD#QUALITY", "3:13", "negative"),
            # ("This place is incredibly tiny.", "place", "RESTAURANT#MISCELLANEOUS", "5:10", "negative"),
            # ("I found the food to be outstanding, particulary the salmon dish I had.", "food", "FOOD#QUALITY", "12:16", "positive"),
            # ("Service was prompt and courteous.", "Service", "SERVICE#GENERAL", "0:7", "positive"),
            # ("delicious bagels, especially when right out of the oven.", "bagels", "FOOD#QUALITY", "10:16", "positive"),
            # ("I have never left a restaurant feeling as if i was abused, and wasted my hard earned money.", "restaurant", "RESTAURANT#GENERAL", "20:30", "negative"),
            ("The food was ok and fair nothing to go crazy.", "food", "FOOD#QUALITY", "4:8", "neutral"),
            # ("overpriced japanese food with mediocre service", "service", "SERVICE#GENERAL", "39:46", "neutral")
        ]

        prompt = self.prompt_instruction
        for ex in examples:
            prompt += (
                f"Sentence: {ex[0]}\n"
                f"Term: {ex[1]}\n"
                f"Aspect category: {ex[2]}\n"
                f"Character offset: {ex[3]}\n"
                f"Sentiment: {ex[4]}\n\n"
            )
        prompt +="\n Analyze and return the sentiment: \n"
        prompt += (
            f"Sentence: {sentence}\n"
            f"Term: {target_term}\n"
            f"Aspect category: {aspect}\n"
            f"Character offset: {offset}\n"
            f"Sentiment:"
        )
        return prompt

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

        y_pred = []
        print("Running ensemble predictions with multiple prompts...")

        for _, row in tqdm(df.iterrows(), total=len(df)):
            predictions = []
            for prompt_builder in self.prompts:
                prompt = prompt_builder(
                    sentence=row["sentence"],
                    target_term=row["target_term"],
                    aspect=row["aspect_category"],
                    offset=row["char_offset"]
                )
                pred = self.call_ollama(prompt)
                predictions.append(pred)

            final = self.majority_vote(predictions)
            y_pred.append(final)

        return y_pred
