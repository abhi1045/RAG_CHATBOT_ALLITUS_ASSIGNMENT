import logging
import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.embeddings import (
    OpenAIEmbeddings,
)
from langchain_community.llms import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.DEBUG)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_OPENAI = os.getenv("USE_OPENAI", "True").lower() == "true"
CHROMA_PATH = "chroma_db"


def get_embeddings():
    if USE_OPENAI and OPENAI_API_KEY:
        return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    else:
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource
def load_rag_chain():
    embeddings = get_embeddings()
    logging.debug(f"Embeddings: {embeddings}")
    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH, embedding_function=embeddings
        )
    except KeyError as e:
        logging.error(f"Chroma initialization failed: {e}")
        return None
    except Exception as e:
        logging.error(f"Error initializing Chroma: {e}")
        raise
    llm = get_llm()
    if llm:
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=False,
        )
        return rag_chain
    else:
        return None


import asyncio


def ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


ensure_event_loop()


def get_llm():
    if USE_OPENAI and OPENAI_API_KEY:
        return OpenAI(openai_api_key=OPENAI_API_KEY)
    else:
        st.warning(
            "OpenAI key not available or USE_OPENAI is False. Chatbot will not be able to answer questions."
        )
        return None


st.title("Customer Support Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you today?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    rag_chain = load_rag_chain()
    if rag_chain:
        result = rag_chain({"query": prompt})
        response = result["result"]
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
    else:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Chatbot is not configured to answer questions without an OpenAI API key or a local LLM.",
            }
        )
        st.chat_message("assistant").write(
            "Chatbot is not configured to answer questions without an OpenAI API key or a local LLM."
        )
