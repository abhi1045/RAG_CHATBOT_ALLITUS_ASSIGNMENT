import os

from dotenv import load_dotenv
from langchain.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import (
    OpenAIEmbeddings,
    SentenceTransformerEmbeddings,
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_OPENAI = (
    os.getenv("USE_OPENAI", "True").lower() == "true"
)  # Default to True if not set
DATA_PATH = "data/support_documentation"
CHROMA_PATH = "chroma_db"


def load_documents(data_path):
    loader_pdf = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    loader_txt = DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader)
    loader_docx = DirectoryLoader(
        data_path, glob="**/*.docx", loader_cls=Docx2txtLoader
    )
    documents = loader_pdf.load() + loader_txt.load() + loader_docx.load()
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vectorstore(chunks, embeddings, persist_directory):
    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory=persist_directory
    )
    vectorstore.persist()
    return vectorstore


def get_embeddings():
    if USE_OPENAI and OPENAI_API_KEY:
        print("Using OpenAI Embeddings.")
        return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    else:
        print("Using Sentence Transformer Embeddings (local).")
        return SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )  # Or another suitable model


if __name__ == "__main__":
    print("Loading documents...")
    documents = load_documents(DATA_PATH)
    print(f"Loaded {len(documents)} documents.")

    print("Splitting documents into chunks...")
    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    print("Creating embeddings...")
    embeddings = get_embeddings()
    vectorstore = create_vectorstore(chunks, embeddings, CHROMA_PATH)
    print(f"Chroma vectorstore saved to {CHROMA_PATH}")
