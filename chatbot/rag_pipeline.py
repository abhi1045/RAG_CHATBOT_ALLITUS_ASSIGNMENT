import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from configs.config import settings

QA_PROMPT = PromptTemplate(
    template=settings.prompt_template, input_variables=["context", "question"]
)


def get_embeddings(use_openai=False):
    if use_openai and settings.openai_api_key:
        print("Using OpenAI Embeddings.")
        return OpenAIEmbeddings(openai_api_key=settings.openai_api_key)
    else:
        print("Using Sentence Transformer Embeddings (local).")
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def load_vectorstore(embeddings, persist_directory):
    try:
        vectorstore = Chroma(
            persist_directory=persist_directory, embedding_function=embeddings
        )
        return vectorstore
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        return None


def get_llm(use_openai=False, local_llm_model_name=settings.local_llm_model_name):
    if use_openai and settings.openai_api_key:
        print("Using OpenAI Chat Model.")
        return ChatOpenAI(
            openai_api_key=settings.openai_api_key, model_name="gpt-3.5-turbo"
        )  # You can choose a different model
    else:
        print(f"Loading local LLM: {local_llm_model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(local_llm_model_name)
            model = AutoModelForCausalLM.from_pretrained(
                local_llm_model_name, device_map="auto"
            )  # Use device_map="auto" to leverage GPU if available

            pipe = pipeline(
                "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256
            )
            llm = HuggingFacePipeline(pipeline=pipe)
            return llm
        except Exception as e:
            st.error(f"Error loading local LLM: {e}")
            return None


def create_rag_chain(llm, vectorstore):
    if llm and vectorstore:
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True,
        )
        return rag_chain
    else:
        print("Local LLM or vectorstore not loaded. RAG chain cannot be created.")
        return None


def query_rag_chain(rag_chain, query):
    if rag_chain:
        result = rag_chain({"query": query})
        return result["result"], result["source_documents"]
    else:
        return "RAG chain not initialized.", []


if __name__ == "__main__":
    use_openai = settings.use_openai
    local_llm_model_name = settings.local_llm_model_name
    embeddings = get_embeddings(use_openai=use_openai)
    vectorstore = load_vectorstore(embeddings, settings.chroma_path)
    llm = get_llm(use_openai=use_openai, local_llm_model_name=local_llm_model_name)
    rag_chain = create_rag_chain(llm, vectorstore)

    while True:
        query = input("Ask your question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        if rag_chain:
            answer, sources = query_rag_chain(rag_chain, query)
            print("Answer:", answer)
            if sources:
                print("Sources:")
                for doc in sources:
                    print(f"- {doc.metadata['source']}")
        else:
            print("Chatbot is not ready.")
