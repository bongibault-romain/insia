import faiss
import ollama
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from transformers import (AutoTokenizer, AutoModel)
from langchain_ollama import OllamaLLM
import time
from PyPDF2 import PdfReader



class KnowledgeBase:
    
    """

    """
    def __init__(self,dataset:list,token_embed_str:str,model_embed_str:str,index_path:str,load:bool=False,method:int = 1):
        start = time.time()
        self.index_path=index_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loadTokeniser(token_embed_str,model_embed_str)
        self.setDataset(dataset)
        if load: self.index = faiss.read_index(index_path)
        elif method == 1 : 
            self.build_faiss_index()
        elif method == 2 : self.make_index_IP(token_embed_str)
        end = time.time()
        print(f"[KnowledgeBase] Temps d'exécution : {end - start:.2f} secondes")

    def loadTokeniser(self,token_embed_str:AutoTokenizer,model_embed_str:AutoModel):
        self.tokenizer_embed = AutoTokenizer.from_pretrained(token_embed_str) # tokenize
        self.model_embed = AutoModel.from_pretrained(model_embed_str).to(self.device) # vectorize
    def build_faiss_index(self):
        start1 = time.time()
        dimension = 384 #vecteur de 384 dimensions pour chaque token
        self.index = faiss.IndexFlatIP(dimension)
        embeddings = np.vstack([self.get_embedding(q["info"]) for q in self.dataset])
        self.index.add(embeddings)
        end1 = time.time()
        faiss.write_index(self.index,self.index_path) 
        end2 = time.time()
        print(f"[build_faiss_index] Temps d'exécution : {end2 - start1:.2f} secondes, avec {end1 - start1:.2f} secondes pour calculer l'index")

    def make_index_IP(self,token_embed:str):
        start1 = time.time()
        faiss_model = SentenceTransformer(token_embed)
        embeddings = np.array([faiss_model.encode(doc["info"]) for doc in self.dataset], dtype=np.float32)
        # FAISS : Créer un index de recherche (cosine similarity)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product = Cosine Similarity si les embeddings sont normalisés
        self.index.add(embeddings)
        end1 = time.time()
        start2 = time.time()
        faiss.write_index(self.index, "faiss_index.idx")
        end2 = time.time()
        print(f"[make_index_IP] Temps d'exécution : {end2 - start1:.2f} secondes, avec {end1 - start1:.2f} secondes pour calculer l'index")

    def setDataset(self,dataset:list):
        self.dataset=dataset
    def addDataset(self,dataset:list):
        self.dataset.append(dataset)

    def get_embedding(self, text):
        inputs = self.tokenizer_embed(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            output = self.model_embed(**inputs)
        return output.last_hidden_state[:, 0, :].cpu().numpy()
    
  

class QueryRewriter:
    def __init__(self,rewrite_model:str):
        pass
    def rewrite(self,query_user:str):
        pass
    def verify(self,query_user:str,query:str):
        pass
    def set_context(self,context:str):
        pass


class QueryExpander:
    """
    Expand Module : used to split a big query in several small ones and verify the subqueries obtained.

    Examples:
        >>> expander=QueryExpander(expander_model='llama3.3')
        >>> expander.expand(query,nb_max_query=5)
        [subquery1,subquery2]
    """
    def __init__(self,expander_model:str):
        self.expander_model=expander_model

    def expand(self,query:str,nb_max_query:int): 
        """
        Split a big query in several small ones.

        Args:
            query (string): The query to split.
            nb_max_query (int): The max number of subqueries returned.

        Returns:
            list: The list of subqueries returned.

        Example:
            >>> expander=QueryExpander(expander_model='llama3.3')
            >>> expander.expand(query,nb_max_query=5)
            [subquery1,subquery2]
        """
        pass
    def verify(self,query:str,queries:list):
        pass
    def setModel(self,expander_model:str):
        self.expander_model=expander_model


class VectorFetcher:
    def __init__(self,knowledge:KnowledgeBase):
        self.knowledge=knowledge
    def retrieve(self,query:str,num_queries=5,date_adjust:bool=True): 
        start = time.time()
        query_embedding = self.knowledge.get_embedding(query)
        D, I = self.knowledge.index.search(query_embedding, k=num_queries)
        retrieved_infos = [self.knowledge.dataset[i] for i in I[0]]
        if VERBOSE>=2: 
            print("question: ", query)
            for i in range(0,num_queries):
                print(f"context: {retrieved_infos[i]["info"]} : score = {D[0][i]:.2f}")
        
        end = time.time()
        print(f"[VectorFetcher] Temps d'exécution : {end - start:.2f} secondes")
        return retrieved_infos 
    def verify(self,query:str,queries:list):
        pass
    


class ChainManager:
    
    class ChainMySQL():
        def retrieve(self,query:str,bdd:str):
            pass
        
    class ChainSem(): 
        """
        A chain to generate queries to question a endpoint OWL/SPARQL

        Examples:
        >>> chainSPARQL=ChainSem(query_endpoint="http://localhost:3030/cluedo/query",model='llama3')
        >>> answer = chainSPARQL.ask("Combien y a-t-il de pièces dans la maison?")
        >>> answer['result']

        'According to the available information, there are 11 pieces in the house.'

        >>> answer['sparql_query']

        PREFIX lamaisondumeurtre: <http://www.lamaisondumeurtre.fr#>
        SELECT (COUNT(?piece) AS ?count)
        WHERE {
            ?house a lamaisondumeurtre:Maison .
            ?house lamaisondumeurtre:pieceDansMaison ?piece .
        }
        
        """
        
        def __init__(self,query_endpoint:str,model:str):
            from langchain_community.graphs import RdfGraph
            from langchain_community.chains.graph_qa.sparql import GraphSparqlQAChain

            self.query_endpoint=query_endpoint
            self.graph = RdfGraph(
                query_endpoint=self.query_endpoint,
                standard="rdf",
                local_copy="test.ttl",
            )
            self.model=model
            llm = OllamaLLM(model=self.model)
            verbose=VERBOSE>=1
            self.chain = GraphSparqlQAChain.from_llm(llm, graph=self.graph, verbose=verbose,allow_dangerous_requests=True, return_sparql_query= True)

        def ask(self,question:str):
            response = self.chain.invoke(question)
            return response



class RAGGenerator:
    def generate(self,query:str,context:str):
        input_text = f"context: {context} question: {query}"

        response = ollama.chat(model='llama3', messages=[
            {
                'role': 'system',
                'content': 'développe ton raisonnement mais n\'invente rien, ne doute jamais du contexte qui t\'est donné, dis clairement si tu ne sais pas la réponse. Le contexte qui t\'est donné est le réglement des études, cite la page d\'origine des informations essentielles. '
            },
            {
                'role': 'user',
                'content': input_text,
            },
        ])

        return response.message.content

class UserPrompt:
    def __init__(self):
        pass
    def ask(self,user_query,nb_contextes):
        start = time.time()
        print("\n\n---------------------------\n",user_query)
        context=fetcher.retrieve(user_query,num_queries=nb_contextes)

        str_context=""
        for i in range(nb_contextes):
            str_context+=context[i]["context"]+str(context[i]["metadata"])+" \n"
        generator=RAGGenerator()
        print(generator.generate(query=user_query,context=str_context))
            
        end = time.time()
        if VERBOSE>=1:print(f"[ask] Temps d'exécution : {end - start:.2f} secondes")
    def askloop(self):
        global VERBOSE
        user_input=None
        stop=(user_input in ("q", "x", "","quit","exit"))
        nb=5
        while not stop:
            user_input=input("Bonjour quelle est votre question ?\n")
            if user_input in ("q", "x", "","quit","exit"):
                break
            elif user_input=="/v":
                v = input("quelle niveau de VERBOSE ? \n")
                while (not v.isnumeric() or int(v)>6 or int(v)<0):
                    print("valeur incorrecte")
                    v = input("quelle niveau de VERBOSE ?\n")
                VERBOSE=int(v)
            elif user_input=="/n":
                nb = input("Combien de contextes donner ?\n")
                while (not nb.isnumeric() or int(nb)>20 or int(nb)<2):
                    print("valeur incorrecte")
                    nb = input("Combien de contextes donner ?\n")
                nb=int(nb)
            else:
                self.ask(user_input,nb_contextes=nb)

    def askfile(self):
        pass




class RAGDataset:
    def extractPDF(self, pdf_path, txt_output_path, meta_output_path):
        from PyPDF2 import PdfReader

        reader = PdfReader(pdf_path)
        full_text = ""
        page_indices = []  # Pour stocker les pages de chaque paragraphe

        all_paragraphs = []

        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if not page_text:
                continue
            lines = page_text.split("\n")
            lines = [line for line in lines if len(line) > 15]

            # Groupe les lignes par 3
            for i in range(0, len(lines), 3):
                paragraph = "".join(lines[i:i+3])
                if len(paragraph.strip()) > 0:
                    all_paragraphs.append(paragraph.strip())
                    page_indices.append(page_num + 1)  # Numérotation des pages commence à 1

        # Écriture du texte
        with open(txt_output_path, "w", encoding="utf-8") as f_txt, \
             open(meta_output_path, "w", encoding="utf-8") as f_meta:
            for paragraph, page in zip(all_paragraphs, page_indices):
                f_txt.write(paragraph + "\n")
                f_meta.write(str(page) + "\n")  # une ligne par paragraphe

    def refineTXT(self, input_path, output_path):
        with open(input_path, "r", encoding="utf-8") as f_in, \
             open(output_path, "w", encoding="utf-8") as f_out:
            for i in f_in:
                f_out.write(refine(i, filtre) + "\n")

    def make_context(self, context_path, refined_path, meta_path, small_to_big):
        # small_to_big=(1,2)
        bef = small_to_big[0]
        aft = small_to_big[1]
        dataset = []

        with open(context_path, 'r', encoding="utf-8") as f_context, \
             open(refined_path, 'r', encoding="utf-8") as f_refined, \
             open(meta_path, 'r', encoding="utf-8") as f_meta:

            context_lines = f_context.readlines()
            refined_lines = f_refined.readlines()
            meta_lines = [int(line.strip()) for line in f_meta.readlines()]

            for i in range(bef, len(context_lines) - aft):
                dataset.append({
                    "info": refined_lines[i],
                    "context": concaten(cut(context_lines, (bef, i, aft))),
                    "metadata": {
                        "page": meta_lines[i]
                    }
                })

        return dataset




def cut(l,param):
    return l[param[1]-param[0]:param[1]+param[2]+1:]
def concaten(l):
    ret=""
    for i in l:
        ret+=i+" "
    return ret
def filtre(string:str):
    for i in string:
        if i.isnumeric():
            return False
        if i.isupper():
            return False
    if len(string)>5:
        return False
    else: return True


def refine(string:str,filtre):
    l=string.split()
    li=[]
    for i in l:
        if not filtre(i.strip()):
            li.append(i.strip())
    return concaten(li)










"""
chainSQL1=ChainManager.ChainSem(query_endpoint="http://localhost:3030/cluedo/query",model='llama3')
chainSQL1.ask("Combien y a-t-il de pièces dans la maison ?") #fuseki-server --update --mem /cluedo
dataset2=[
    {"info": "Les registres du processeur XYZ ont une taille de 68 bits.", "date": "2022-01-01", "isChained":False, "chain":None},
    {"info": "Le processeur XYZ possède 8 cœurs physiques et 16 threads.", "date": "2022-01-01", "isChained":True, "chain":chainSQL1}
    ]

"""




VERBOSE=1

if __name__=="__main__":
    
    VERBOSE=2

    start = time.time()

    
    
    
    dataset3=[]
    small_to_big = (1,2)
    RAGDataset().extractPDF("Reglement_des_Etudes_2023-2024.pdf", "reglement.txt","meta.txt")
    RAGDataset().refineTXT("reglement.txt","refined.txt")
    dataset3=RAGDataset().make_context("reglement.txt","refined.txt","meta.txt",(1,2))


    knowledge = KnowledgeBase(dataset3,"BAAI/bge-small-en","BAAI/bge-small-en",index_path="faiss_index.idx",load=True,method=1)
    fetcher=VectorFetcher(knowledge)
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device : ",device)

    
    end = time.time()
    print(f"[global init] Temps d'exécution : {end - start:.2f} secondes")

    user=UserPrompt()
    user.askloop()




