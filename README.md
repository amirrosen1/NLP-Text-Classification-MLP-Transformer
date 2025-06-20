# 🧠 NLP Text Classification

Text classification over 20Newsgroups using different model architectures:  
✅ Log-linear (Single-layer MLP)  
✅ Multi-layer MLP  
✅ Pretrained Transformer (DistilRoBERTa)

---

## 📚 Task Overview

We classify documents from 4 topics in the 20Newsgroups dataset:
- `comp.graphics`
- `rec.sport.baseball`
- `sci.electronics`
- `talk.politics.guns`

We compare performance on different portions of the training set (10%, 20%, 50%, 100%).

---

## 🧪 Models Used

1. **Log-Linear MLP** – single layer using TF-IDF features  
2. **Multi-layer MLP** – with one hidden layer of size 500  
3. **Transformer** – DistilRoBERTa fine-tuned on small subsets (10%, 20%)

---

## 📈 Evaluation

- Training loss and validation accuracy plotted per epoch
- Accuracy compared across models and dataset sizes
- Number of trainable parameters extracted for each model

---

## 📦 Files

| File                 | Description                               |
|----------------------|-------------------------------------------|
| `ex2.py`             | Main code – model definitions and training |
| `requirements.txt`   | List of required Python packages           |
| `Ex2 - Answers.pdf`  | Reported results (ignored by `.gitignore`)|
| `Exercise 2.pdf`     | Instructions for the assignment            |

---

## 🛠️ Install Requirements

```bash
pip install -r requirements.txt
