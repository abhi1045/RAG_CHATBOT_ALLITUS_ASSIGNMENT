import os

from configs.config import settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings


def load_documents(data_path):
    if not os.path.exists(data_path) or not os.listdir(data_path):
        print(
            f"Warning: Directory '{data_path}' does not exist or is empty. No documents loaded."
        )
        return []
    loader_pdf = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    loader_txt = DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader)
    loader_docx = DirectoryLoader(
        data_path, glob="**/*.docx", loader_cls=Docx2txtLoader
    )
    documents = []
    try:
        documents.extend(loader_pdf.load())
    except Exception as e:
        print(f"Error loading PDF documents: {e}")
    try:
        documents.extend(loader_txt.load())
    except Exception as e:
        print(f"Error loading TXT documents: {e}")
    try:
        documents.extend(loader_docx.load())
    except Exception as e:
        print(f"Error loading DOCX documents: {e}")
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def get_embeddings(use_openai=False):
    if use_openai and settings.openai_api_key:
        print("Using OpenAI Embeddings for Ingestion.")
        return OpenAIEmbeddings(openai_api_key=settings.openai_api_key)
    else:
        print("Using Sentence Transformer Embeddings (local) for Ingestion.")
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def create_vectorstore(chunks, embeddings, persist_directory):
    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory=persist_directory
    )
    vectorstore.persist()
    return vectorstore


if __name__ == "__main__":
    print("Loading documents...")
    documents = load_documents(settings.data_path)
    print(f"Loaded {len(documents)} documents.")

    if documents:
        print("Splitting documents into chunks...")
        chunks = split_documents(documents)
        print(f"Split into {len(chunks)} chunks.")

        print("Creating embeddings...")
        embeddings = get_embeddings(use_openai=settings.use_openai)
        vectorstore = create_vectorstore(chunks, embeddings, settings.chroma_path)
        print(
            f"Chroma vectorstore saved to {settings.chroma_path} using {'OpenAI' if settings.use_openai else 'local'} embeddings."
        )
    else:
        print("No documents to process. Skipping vectorstore creation.")
