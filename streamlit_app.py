import os
from dotenv import load_dotenv
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
import re

st.set_page_config(
    page_title="CELia - Assistante INSA",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
)

st.image("logo-insa.png", width=110)

load_dotenv()

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner("Chargement des données pédagogiques..."):
        loader = TextLoader('regetude.txt')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        docs = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        faiss_index = FAISS.from_documents(docs, embeddings)
        return faiss_index

faiss_index = load_data()

llm = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50},
    huggingfacehub_api_token="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"
)

template = """
Tu t'appelles CÉLia. Tu es une assistante francophone de l'INSA de Toulouse. Tu réponds toujours en français, même si la question est posée dans une autre langue.
Tu peux répondre aussi bien à des questions pédagogiques qu'à des questions de conversation générale comme "ça va ?", "tu fais quoi ?", etc.
Utilise le contexte ci-dessous si nécessaire pour répondre à la question. Si tu ne sais pas, dis-le simplement.
Ta réponse doit être concise, naturelle, et tenir en 2 phrases maximum.

Contexte : {context}
Question : {question}
Réponse :
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])
qa_chain_prompt = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
qa_chain = RetrievalQA(retriever=faiss_index.as_retriever(), combine_documents_chain=qa_chain_prompt)

# Streamlit
st.title("CÉLia - Assistante IA de l'INSA 💬✨")
st.info(
    "Je suis CÉLia, votre assistante IA à l'INSA de Toulouse. "
    "Posez-moi vos questions !",
    icon="ℹ️"
)

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour ! Je suis CÉLia. Posez-moi une question sur l'INSA ou discutez avec moi ! :) "}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Votre question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Je réfléchis..."):
            result = qa_chain({"query": prompt})
            raw_output = result["result"]
            
            # Nettoyage de la réponse
            pattern = rf"Question\s*:\s*{re.escape(prompt)}\s*Réponse\s*:\s*(.*?)(?:\nQuestion\s*:|\Z)"
            match = re.search(pattern, raw_output, re.DOTALL | re.IGNORECASE)
            
            if match:
                response = match.group(1).strip()
            else:
                response = "Désolée, je n'ai pas compris la réponse."
            
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
