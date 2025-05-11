

import json
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import os


class RagVerifier():

    def __init__(self,data_path:str="train.json",load=False):

        if load:
            loaded = self.load("verifier_model")
            self.model = loaded.model
            self.model_name = loaded.model_name
            self.classifieur = loaded.classifieur
            self.exemples = None
            return
        else:
            self.exemples = self.load_conditionalqa(data_path, max_items=1500)

            print("Chargement du modèle BGE...")
            
            self.model_name = "BAAI/bge-small-en"
            self.model = SentenceTransformer(self.model_name)

            print("Encodage des exemples...")
            X, y = self.encoder_exemples(self.model, self.exemples)

            print("Entraînement du classifieur...")
            self.classifieur = self.entrainer_classifieur(X, y)
            self.save("verifier_model")
        

    def load_conditionalqa(self,file_path, max_items=1000):
        with open("dataset/ConditionalQA/v1_0/train.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = []
        pos, neg = 0, 0
        for article in data:
            question = article["question"]
            context = article["scenario"]
            answers = article["answers"]
            is_sufficient = all(len(a[1]) == 0 for a in answers)
            label = int(is_sufficient)
            
            # équilibrage artificiel 50/50
            if label == 1 and pos < max_items // 2:
                samples.append((question, context, label))
                pos += 1
            elif label == 0 and neg < max_items // 2:
                samples.append((question, context, label))
                neg += 1
            
            if pos + neg >= max_items:
                break
        return samples

    # === 2. Encodage avec SentenceTransformer (BGE) ===

    def encoder_exemples(self,model, exemples):
        textes = [f"Question: {q} Context: {c}" for q, c, _ in exemples]
        embeddings = self.model.encode(textes, normalize_embeddings=True)
        labels = [label for _, _, label in exemples]
        return np.array(embeddings), np.array(labels)

    # === 3. Entraînement d’un classifieur binaire ===

    def entrainer_classifieur(self,X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))
        return clf


    

    # Exemple d'utilisation
    def verifier_contextes(self,question, contextes):
        texte = f"Question: {question} Context: {' '.join(contextes)}"
        emb = self.model.encode([texte], normalize_embeddings=True)
        return "suffisant" if self.classifieur.predict(emb)[0] == 1 else "insuffisant"
    def save(self, path="verifier_model"):
        import os
        import joblib

        os.makedirs(path, exist_ok=True)
        
        # Sauvegarde du classifieur
        joblib.dump(self.classifieur, os.path.join(path, "xgb_classifier.joblib"))
        # Sauvegarde du nom du modèle de SentenceTransformer
        with open(os.path.join(path, "model_config.json"), "w") as f:
             json.dump({"model_name": self.model_name}, f)
    @classmethod
    def load(cls, path="verifier_model"):
        import joblib
        with open(os.path.join(path, "model_config.json")) as f:
            config = json.load(f)

        instance = cls.__new__(cls)
        instance.model_name = config["model_name"]
        instance.model = SentenceTransformer(instance.model_name)
        instance.classifieur = joblib.load(os.path.join(path, "xgb_classifier.joblib"))
        instance.exemples = None
        return instance

    


# === MAIN ===    

if __name__ == "__main__":
    queries_path="./experiment/queries.txt"
    queries=open(queries_path, 'r', encoding="utf-8").readlines()
    L=[]
    verifier=RagVerifier("train.json",load=False)
    for i in range(20):
        contexts=open(f"./experiment/{i}/contexts.txt", 'r', encoding="utf-8").readlines()
        L.append(verifier.verifier_contextes(queries[i],contexts))
        print(i," : ",queries[i]," : ",L[-1])
    print(L)
    verifier=RagVerifier("train.json",load=True)
    for i in range(20):
        contexts=open(f"./experiment/{i}/contexts.txt", 'r', encoding="utf-8").readlines()
        L.append(verifier.verifier_contextes(queries[i],contexts))
        print(i," : ",queries[i]," : ",L[-1])
    print(L)


