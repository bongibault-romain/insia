from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "cross-encoder/nli-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def score(q, c):
    inputs = tokenizer(q, c, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.softmax(logits, dim=1)[0][1].item()  # probabilité que la paire soit "entailment"

# seuil à calibrer
print("Score (1 = suffisant) :", score("Can I apply?", "You can apply if you are 18 or over..."))
