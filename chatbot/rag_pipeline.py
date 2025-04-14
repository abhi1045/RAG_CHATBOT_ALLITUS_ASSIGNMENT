import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_OPENAI = os.getenv("USE_OPENAI", "False").lower() == "true"
CHROMA_PATH = "chroma_db"


def get_embeddings():
    if USE_OPENAI and OPENAI_API_KEY:
        return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    else:
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def load_vectorstore(embeddings, persist_directory):
    vectorstore = Chroma(
        persist_directory=persist_directory, embedding_function=embeddings
    )
    return vectorstore


def get_llm():
    if USE_OPENAI and OPENAI_API_KEY:
        print("Using OpenAI Chat Model.")
        return ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    else:
        print(
            "OpenAI key not available or USE_OPENAI is False. RAG functionality will be limited to retrieval only."
        )
        return None  # Or integrate with a local LLM here if desired


def create_rag_chain(llm, vectorstore):
    if llm:
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=False,
        )
        return rag_chain
    else:
        return None


def query_rag_chain(rag_chain, query):
    if rag_chain:
        result = rag_chain({"query": query})
        return result["result"]
    else:
        return "LLM not available. Cannot answer queries."


if __name__ == "__main__":
    embeddings = get_embeddings()
    vectorstore = load_vectorstore(embeddings, CHROMA_PATH)
    llm = get_llm()
    rag_chain = create_rag_chain(llm, vectorstore)

    while True:
        query = input("Ask your question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        answer = query_rag_chain(rag_chain, query)
        print("Answer:", answer)
