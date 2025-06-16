import os
import streamlit as st

from decouple import config
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from functions import *

os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')
persist_directory = 'db'

vector_store = load_existing_vector_store(persist_directory)



st.set_page_config(
    page_title="Chatbot PyGPT",
    page_icon=":robot_face:",
)

st.header('Chatbot com seus documentos (RAG)')

with st.sidebar:
    st.header('Upload de arquivos')
    uploaded_files = st.file_uploader(
        label='Upload dos arquivos PDF',
        type=['pdf'],
        accept_multiple_files=True,
        help='Selecione os arquivos PDF que deseja carregar:',
    )
    # Ação para se o usuário fez upload de arquivos
    if uploaded_files:
        with st.spinner('Processando arquivos...'):
            all_chunks = []
            for uploaded_file in uploaded_files:
                chunks = process_pdf(file=uploaded_file)
                all_chunks.extend(chunks)
            vector_store = add_to_vector_store(
                chunks=all_chunks,
                vector_store=vector_store,
                persist_directory=persist_directory,
            )
            

    model_options = {
        'GPT-4o': 'gpt-4o',
        'GPT-4o Mini': 'gpt-4o-mini',
        'GPT-4.5 Preview': 'gpt-4.5-preview',
        'O4 Mini': 'o4-mini',        
    }

    selected_model = st.sidebar.selectbox(
        label='Selecione o modelo LLLM',
        options=model_options,
        index=0,
        help='Selecione qual modelo de LLM você deseja usar:',
    )

    st.divider()
    st.sidebar.markdown('###### Criado por:')
    st.sidebar.markdown('###### [@marcelo_borges](https://www.linkedin.com/in/mborgesx/)')

# Aqui está sendo criado uma lista vazia para armazenar as mensagens do usuário e do chatbot
if 'messages' not in st.session_state:
    st.session_state.['message'] = []

question = st.chat_input('Digite sua pergunta aqui:')

if vector_store