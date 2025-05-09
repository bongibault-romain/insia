from INSight import RAGDataset, KnowledgeBase, VectorFetcher, RAGGenerator
import config
import time
import os

def expe(refined,chunk_size):
    dataset3=[]


    path="./experiment"
    dataset3=RAGDataset(load=False,data_path=path,chunk_size=chunk_size)


    kb=KnowledgeBase(dataset3,config.EMBED_MODEL,config.EMBED_MODEL)

    start = time.time()
    kb.build_faiss_index(refined=refined)
    
    end = time.time()
    duration=round(end-start,2)
    index=kb.index

    # Recherche des 2 plus proches voisins de chaque vecteur du jeu de base
    import numpy as np
    import faiss


    D, I = index.search(kb.embeddings, k=2)  # distances et indices des 2 plus proches voisins

    distances=np.zeros(len(kb.embeddings))
    # Affichage
    count=0
    for i in range(len(kb.embeddings)):
        self_index = I[i][0]  # devrait être i
        neighbor_index = I[i][1]  # plus proche voisin autre que lui-même
        distance = D[i][1]  # similarité cosinus car IndexFlatIP
        distances[i]=distance
        if distance>0.97:
            count+=1
    entry={}
    entry["refined"]=refined
    entry["chunk_size"]=chunk_size

    entry["mean"]=np.mean(distances)
    entry["max"]=np.max(distances)
    entry["min"]=np.min(distances)
    entry["median"]=np.median(distances)
    entry[">97%"]=round(count/len(kb.embeddings)*100,2)
    entry["variance"]=np.nanvar(distances)
    entry["duration"]=duration
    import utils
    utils.save_entry(entry)


def grid_expe():

    from sklearn.model_selection import ParameterGrid
    param_grid = {'refined': [True,False], 'chunk_size': [20, 30, 40, 50, 60, 70,80,100,120,150,180,220,250,300,350,400,450,500,600]}





    params=ParameterGrid(param_grid)

    for param in params : 
        expe(**param)


    import pandas as pd
    import numpy as np
    import utils
    data = utils.load_entries()


    table = pd.DataFrame.from_dict(data)
    table = table.replace(np.nan, '-')
    table = table.sort_values(by='max', ascending=False)

    from IPython.display import display

    display(table)


refined = True
chunk_size = 120
dataset3=[]


path="./experiment"
dataset3=RAGDataset(load=False,data_path=path,chunk_size=chunk_size)


kb=KnowledgeBase(dataset3,config.EMBED_MODEL,config.EMBED_MODEL)

start = time.time()
#kb.build_faiss_index(refined=refined)

end = time.time()
duration=round(end-start,2)

fetcher=VectorFetcher(kb)


queries_path="./experiment/queries.txt"
queries=open(queries_path, 'r', encoding="utf-8").readlines()


'''
for i in range(len(queries)):
    contexts=fetcher.retrieve(queries[i].strip())
    if not os.path.exists(f"./experiment/{i}/"):
        os.mkdir(f"./experiment/{i}/")
    with open(f"./experiment/{i}/contexts.txt","w", encoding="utf-8") as f:
        for ctx in contexts:
            f.write(str(ctx)+"\n")


'''
def verif(query,contexts):
    start = time.time()
    from ollama import chat
    context=""
    for i in contexts:
        context+=i
    input_text = f"context: {context} question: {query}"

    response = chat(model="llama3:latest", messages=[
        {
            'role': 'system',
            'content': (
                "Tu es un classifieur. Ta tâche est de répondre uniquement par \"Oui\" ou \"Non\".\n"
                "Réponds \"Oui\" **uniquement** si les contextes sont suffisants pour répondre "
                "à la question de façon claire et sans ambiguïté. Sinon, réponds \"Non\".\n"
                "Ta réponse **doit** être uniquement \"Oui\" ou \"Non\", sans aucune explication ni autre mot."
            )
        },
        {
            'role': 'user',
            'content': input_text,
        },
    ])
    end = time.time()
    print(f"[RAGGenerator] Temps d'exécution : {end - start:.2f} secondes")
    return response.message.content

print("Gen : ...")
i = 12
contexts=open(f"./experiment/{i}/contexts.txt", 'r', encoding="utf-8").readlines()
rg=RAGGenerator()
rep=rg.generate(queries[i],contexts)
print(rep)