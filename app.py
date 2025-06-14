import os
import streamlit as st

from decouple import config

os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')

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