import tempfile
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Função para receber os arquivo, porém em binário, então vamos salter em tempfile para converter 
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    
    os.remove(temp_file_path)

    # Importado o arquivo recebido, vamos quebrar em chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,
    )
    chunks = text_splitter.split_documents(documents=docs)
    return chunks