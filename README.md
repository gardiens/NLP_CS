# 🤖 NLP CS Project

## 👥 Contributors
The following students contributed to this project:
- Antonine Batifol  
- Pierrick Bournez  
- Gabrielle Caillaud  


## 🧠 Classifier Description
We implemented an *in-context* classification approach using few-shot learning. To mitigate prediction bias, we designed multiple prompts with varying polarity.  

After evaluating several models, we selected **Gemma 3:4B** as the backbone due to its favorable trade-off between performance and inference time.  

For each input sample, the model was queried for its polarity based on a set of carefully chosen examples. The key specificity of our method lies in the use of **three distinct prompts**:
1. One where the examples are labeled as **positive**
2. One where the examples are labeled as **neutral**
3. One where the examples are labeled as **negative**

By averaging the outputs across these prompts, we aimed to reduce ambiguity and enhance the robustness of the polarity prediction.
## 📊 Accuracy on the Development Set
On the development set, our classifier achieved an accuracy of:  
**91.87 ± 0.27**