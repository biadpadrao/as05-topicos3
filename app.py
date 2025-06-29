import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HUGGINGFACE_TOKEN")
if not token:
    st.error("Configure a variável HUGGINGFACE_TOKEN no ambiente.")
    st.stop()

#   Modelos usados para embeddings e LLM
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
llm_model_name = "HuggingFaceH4/zephyr-7b-beta"

#   Função para extrair texto dos PDFs
def extract_text_from_pdfs(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

st.set_page_config(page_title="Perguntas sobre PDF", layout="wide")
st.title("Assistente Conversacional LLM para PDFs")
st.subheader("Baseado na API da Hugging Face, para textos em pt-br")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

uploaded_pdfs = st.file_uploader("Envie seus arquivos PDF", type="pdf", accept_multiple_files=True)

#   Processamento dos PDFs e criação do vectorstore
if st.button("Processar PDFs"):
    if uploaded_pdfs:
        raw_text = extract_text_from_pdfs(uploaded_pdfs)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(raw_text)

        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
        )
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)

        st.session_state.vector_store = vector_store
        st.success(f"PDFs processados e indexados em {len(chunks)} pedaços.")
    else:
        st.warning("Por favor, envie pelo menos um arquivo PDF.")

#   Entrada da pergunta
user_question = st.chat_input("Faça sua pergunta sobre os documentos:")

#   Gerar resposta usando RetrievalQA com LLM do HuggingFaceHub
if user_question:
    if "vector_store" not in st.session_state:
        st.warning("Faça o upload e processamento dos PDFs primeiro.")
    else:
        llm = HuggingFaceHub(
            repo_id=llm_model_name,
            huggingfacehub_api_token=token,
            model_kwargs={"temperature": 0.1, "max_new_tokens": 512}
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="map_reduce", 
            retriever=st.session_state.vector_store.as_retriever()
        )

        with st.spinner("Buscando resposta..."):
            answer = qa_chain.run(user_question)

        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
