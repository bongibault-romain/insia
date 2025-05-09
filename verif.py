#dataset\ConditionalQA\evaluate.py

import os
import json
import requests
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np




def load_conditionalqa(file_path, max_items=1000):
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

def encoder_exemples(model, exemples):
    textes = [f"Question: {q} Context: {c}" for q, c, _ in exemples]
    embeddings = model.encode(textes, normalize_embeddings=True)
    labels = [label for _, _, label in exemples]
    return np.array(embeddings), np.array(labels)

# === 3. Entraînement d’un classifieur binaire ===

def entrainer_classifieur(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    from xgboost import XGBClassifier
    clf = XGBClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    return clf

# === MAIN ===

if __name__ == "__main__":
    exemples = load_conditionalqa("train.json", max_items=1500)

    print("Chargement du modèle BGE...")
    model = SentenceTransformer("BAAI/bge-small-en")

    print("Encodage des exemples...")
    X, y = encoder_exemples(model, exemples)

    print("Entraînement du classifieur...")
    classifieur = entrainer_classifieur(X, y)

    # Exemple d'utilisation
    def verifier_contextes(question, contextes):
        texte = f"Question: {question} Context: {' '.join(contextes)}"
        emb = model.encode([texte], normalize_embeddings=True)
        return "suffisant" if classifieur.predict(emb)[0] == 1 else "insuffisant"

    # Test simple
    print("\nTest :")
    test_q = "What is photosynthesis?"
    test_c = ["Photosynthesis allows plants to convert sunlight into chemical energy."]
    print(verifier_contextes(test_q, test_c))
    test_q = "Quel est le but du règlement des études ?"
    test_c = ["""Le règlement des études a pour objet de définir les règles en vigueur dans le domaine de la formation
    et d’informer la communauté universitaire.
    Le règlement des études est opposable à toute personne, enseignant ou étudiant à l’INSA, qui de
    facto, l’accepte en rejoignant l’établissement."""]
    print(verifier_contextes(test_q, test_c))
    test_q = "Quelles sont les règles du règlement des études ?"
    test_c = ["""Le règlement des études a pour objet de définir les règles en vigueur dans le domaine de la formation
    et d’informer la communauté universitaire.
    Le règlement des études est opposable à toute personne, enseignant ou étudiant à l’INSA, qui de
    facto, l’accepte en rejoignant l’établissement."""]
    print(verifier_contextes(test_q, test_c))
    test_q = "Combien de formations à l'INSA ?"
    test_c = ["""Le règlement des études a pour objet de définir les règles en vigueur dans le domaine de la formation
    et d’informer la communauté universitaire.
    Le règlement des études est opposable à toute personne, enseignant ou étudiant à l’INSA, qui de
    facto, l’accepte en rejoignant l’établissement."""]
    print(verifier_contextes(test_q, test_c))
    test_q = "Quelle est la taille de l'INSA"
    test_c = ["""Le règlement des études a pour objet de définir les règles en vigueur dans le domaine de la formation
    et d’informer la communauté universitaire.
    Le règlement des études est opposable à toute personne, enseignant ou étudiant à l’INSA, qui de
    facto, l’accepte en rejoignant l’établissement."""]
    print(verifier_contextes(test_q, test_c))
    test_q = "HDBCGFEFBF ?"
    test_c = ["""Le règlement des études a pour objet de définir les règles en vigueur dans le domaine de la formation
    et d’informer la communauté universitaire.
    Le règlement des études est opposable à toute personne, enseignant ou étudiant à l’INSA, qui de
    facto, l’accepte en rejoignant l’établissement."""]
    print(verifier_contextes(test_q, test_c))


    

    queries_path="./experiment/queries.txt"
    queries=open(queries_path, 'r', encoding="utf-8").readlines()
    L=[]
    for i in range(20):
        contexts=open(f"./experiment/{i}/contexts.txt", 'r', encoding="utf-8").readlines()
        L.append(verifier_contextes(queries[i],contexts))
        print(i," : ",queries[i]," : ",L[-1])
    print(L)