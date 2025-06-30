import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

token = os.getenv("HUGGINGFACE_TOKEN")
if not token:
    st.error("Configure a variável HUGGINGFACE_TOKEN no ambiente.")
    st.stop()

# Modelos usados para embeddings e LLM
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
llm_model_name = "HuggingFaceH4/zephyr-7b-beta"

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

# Upload de PDFs
uploaded_pdfs = st.file_uploader("Envie seus arquivos PDF", type="pdf", accept_multiple_files=True)

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

# Definir template prompt 
prompt_template = """
Use o contexto a seguir para responder à pergunta de forma concisa e direta. Não inclua o contexto na resposta, apenas a informação solicitada.

Contexto: {context}

Pergunta: {question}

Resposta:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Entrada da pergunta
user_question = st.chat_input("Faça sua pergunta sobre os documentos:")

# Gerar resposta usando RetrievalQA com LLM do HuggingFaceHub
if user_question:
    if "vector_store" not in st.session_state:
        st.warning("Faça o upload e processamento dos PDFs primeiro.")
    else:
        try:
            llm = HuggingFaceHub(
                repo_id=llm_model_name,
                huggingfacehub_api_token=token,
                model_kwargs={"temperature": 0.01, "max_new_tokens": 100}
            )

            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 1})

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt}
            )

            with st.spinner("Buscando resposta..."):
                response = qa_chain.invoke({"query": user_question})
                answer = response.get("result", "Erro ao obter a resposta.")

            with st.chat_message("user"):
                st.markdown(user_question)
            st.session_state.messages.append({"role": "user", "content": user_question})

            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Erro inesperado: {str(e)}")
            st.info("Consulte os logs para mais detalhes ou tente novamente.")