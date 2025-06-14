import os
import streamlit as st

from decouple import config
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from functions import process_pdf

os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')
persist_directory = 'db'






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
            print(all_chunks)

    model_options = [
        'gpt-3.5-turbo',
        'gpt-4',
        'gpt-4o',       
        'gpt-4o-mini',
        'gpt-4-turbo', 
    ]

    selected_model = st.sidebar.selectbox(
        label='Selecione o modelo LLLM',
        options=model_options,
        index=0,
        help='Selecione qual modelo de LLM você deseja usar:',
    )

    st.divider()
    st.sidebar.markdown('###### Criado por:')
    st.sidebar.markdown('###### [@marcelo_borges](https://www.linkedin.com/in/mborgesx/)')
    

question = st.chat_input('Digite sua pergunta aqui:')

st.chat_message('user').write(question)