import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente do .env (ou configure pelo Secrets do Streamlit Cloud)
load_dotenv()
token = os.getenv("HUGGINGFACE_TOKEN")
if not token:
    st.error("Configure a variável HUGGINGFACE_TOKEN nos Secrets do Streamlit Cloud.")
    st.stop()

# Modelos usados
embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
llm_model = "google/flan-t5-base"  

def extract_text_from_pdfs(pdfs):
    texts = []
    for pdf in pdfs:
        reader = PdfReader(pdf)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        texts.append(text)
    return texts

st.set_page_config(page_title="Perguntas sobre PDF", layout="wide")
st.title("Assistente Conversacional LLM para PDFs")
st.subheader("Baseado na API da Hugging Face, para textos em pt-br")

uploaded_pdfs = st.file_uploader("Envie seus PDFs", type="pdf", accept_multiple_files=True)

if st.button("Processar PDFs"):
    if uploaded_pdfs:
        raw_texts = extract_text_from_pdfs(uploaded_pdfs)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = []
        for text in raw_texts:
            chunks.extend(splitter.split_text(text))

        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        vector_store = FAISS.from_texts(chunks, embedding=embeddings)

        st.session_state["vector_store"] = vector_store
        st.success(f"Processados {len(uploaded_pdfs)} PDFs e criados {len(chunks)} chunks.")
    else:
        st.warning("Por favor, envie pelo menos um arquivo PDF.")

question = st.text_input("Faça sua pergunta sobre os documentos:")

if question:
    if "vector_store" not in st.session_state:
        st.warning("Primeiro faça o upload e processamento dos PDFs.")
    else:
        # Criar o LLM usando token e configurando a tarefa correta
        llm = HuggingFaceHub(
            repo_id=llm_model,
            huggingfacehub_api_token=token,
            task="text-generation",
            model_kwargs={"temperature": 0.1, "max_new_tokens": 256},
        )

        # Criar cadeia de QA com recuperação
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state["vector_store"].as_retriever()
        )

        with st.spinner("Buscando resposta..."):
            answer = qa.run(question)

        st.markdown(f"**Resposta:** {answer}")
