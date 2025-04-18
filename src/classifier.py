from typing import List
from utils import load_data, remap_aspect, system_prompt
import torch
from ollama import Client


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
      self.model = None
      self.client = Client(
          host=ollama_url
      )
     
    
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
    # in-context learning with ollama
    pass



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

    test_data = load_data(data_filename)
    test_data = test_data.map(remap_aspect)

    predicted_labels = []

    for i in range(len(test_data)):
      current_user_prompt = f'''
                              Sentence: {test_data[i]['Sentence']}
                              Aspect: {test_data[i]['Aspect_Category']}, Term: {test_data[i]['Target_term']}
                              Sentiment: '''

      response = self.client.chat(model='gemma3:1b', messages=[
                                                                {
                                                                  'role': 'system',
                                                                  'content': system_prompt,
                                                                },
                                                                {'role': 'user', 
                                                                'content': current_user_prompt}
                                                                ])

      classification = response.message.content.split("Sentiment:")[-1] # take the part after Sentiment
      if "positive" in classification:
        predicted_labels.append("positive")
      elif "negative" in classification:
        predicted_labels.append("negative")
      else:
        print(f"undefined. using positive per default. Response from model: {response.message.content}")
        predicted_labels.append("positive")
    return predicted_labels



