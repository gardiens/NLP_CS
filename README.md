# 🤖 NLP CS Project

## 👥 Contributors
The following students contributed to this project:
- Antonine Batifol  
- Pierrick Bournez  
- Gabrielle Caillaud  


## 🧠 Classifier Description
We implemented an *in-context* classification approach using few-shot learning. To mitigate prediction bias and enhance robustness, we designed multiple prompts with varying polarity.  The design of the prompt is described in Section [Prompting Strategy](#prompting-strategy-and-other-experiments) 

After evaluating several models, we selected **Gemma 3:4B** as the backbone due to its favorable trade-off between performance and inference time.  

For each input sample, the model was queried for its polarity based on a set of carefully chosen examples. The key specificity of our method lies in the use of **three distinct prompts**:
1. One where the example is labeled as **positive**
2. One where the example is labeled as **neutral**
3. One where the example is labeled as **negative**

Each example includes the sentence, the aspect term with its offset, the aspect category and its polarity. The prompt first includes some brief instruction followed by the example and the task input.

By averaging the outputs across these prompts, we aimed to reduce ambiguity and enhance the robustness of the polarity prediction.

## 📊 Accuracy on the Development Set
On the development set, our classifier achieved an accuracy of:  
**91.87 ± 0.27**

## Prompting strategy and other experiments

**Prompting Strategy:**
Our prompting strategy is based on a polarity-conditioned one-shot  prompt.
- **We constructed the prompt with concise and minimal instructions**. Longer task instructions seemed to confuse the model and reduce accuracy likely because we used smaller quantized model.
- **We use one-shot prompts with  three different polarity to reduce bias and maximize its performance**. We also tried the same approach with unbalanced few-shots with 2 neutral, 1 postive, 1 negative / 1 neutral, 2 positive, 1 negative and 1 neutral, 1 positive, 2 negative but the accuracy was lower.
- **We limited ensemble size to three prompts** . We wanted to balance accuracy and inference speed, since inference time was also part of the evaluation criteria.

**Other Exploration**
We experimented with various possible enhancements:
- Role-based prompting and long-format task guidelines, however long prompts reduced accuracy maybe due to loss of attention on key input features.
- Using special tokens [T]...[/T] to emphasize the target term in context instead of the offset, which did not improve our accuracy
- Dynamic retrieval of few-shot examples using semantic similarity via BERT embeddings. We embedded the entire train set during training, and then compute the similiraty between the test instance and the train set to retrieve the 3 most similar. We include these 3 examples in our few-shots prompt. Embedding-based dynamic retrieval was promising but increased latency (it took 3 minutes to embed the train set and it also increased inference time due to the retrieval computation) and complexity without consistent accuracy gains.

