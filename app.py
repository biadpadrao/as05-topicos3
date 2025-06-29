import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
import os

load_dotenv()

HUGGINGFACE_TOKEN_AS05_API = os.getenv("HUGGINGFACE_TOKEN_AS05_API")
if not HUGGINGFACE_TOKEN_AS05_API:
    st.error("Configure a variável HUGGINGFACE_TOKEN_AS05_API no arquivo .env")
    st.stop()

embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
llm_model_name = "pierreguillou/bert-base-cased-squad-v1.1-portuguese" 

embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, 
                                   huggingfacehub_api_token=HUGGINGFACE_TOKEN_AS05_API)

llm = HuggingFaceHub(repo_id=llm_model_name, 
                     huggingfacehub_api_token=HUGGINGFACE_TOKEN_AS05_API)

st.set_page_config(page_title="Perguntas sobre PDF", layout="wide")
st.title("Assistente Conversacional LLM para PDFs")
st.subheader("Baseado na API da Hugging Face, para textos em pt-br")

uploaded_file = st.file_uploader("Envie seu arquivo PDF", type=["pdf"])

if uploaded_file:
    # Carregar PDF
    loader = PyPDFLoader(uploaded_file)
    documents = loader.load()

    # Dividir texto em chunks menores para embeddings
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(documents)

    # Indexar com FAISS
    with st.spinner("Gerando embeddings e indexando documentos..."):
        vectorstore = FAISS.from_documents(texts, embeddings)

    st.success(f"{len(texts)} pedaços indexados!")

    question = st.text_input("Digite sua pergunta sobre o documento:")

    if question:
        with st.spinner("Buscando resposta..."):
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
            answer = qa.run(question)
        st.subheader("Resposta:")
        st.write(answer)
