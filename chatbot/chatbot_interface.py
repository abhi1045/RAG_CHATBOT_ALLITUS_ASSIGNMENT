import streamlit as st
import torch
from configs.config import settings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.classes.__path__ = []

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


@st.cache_resource
def load_retriever(use_openai_embeddings=False):
    embeddings = get_embeddings(use_openai_embeddings)
    vectorstore = Chroma(
        persist_directory=settings.chroma_path, embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})


@st.cache_resource
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


st.title("Local/OpenAI Customer Support Chatbot")
st.subheader("Answers are generated locally or via OpenAI.")

use_openai_api = st.checkbox(
    "Use OpenAI API (requires API key)", value=settings.use_openai
)
local_llm_model = st.text_input(
    "Local LLM Model Name", value=settings.local_llm_model_name
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you today?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    retriever = load_retriever(use_openai_embeddings=use_openai_api)
    llm = get_llm(use_openai=use_openai_api, local_llm_model_name=local_llm_model)

    if retriever and llm:
        relevant_documents = retriever.get_relevant_documents(prompt)
        if not relevant_documents:
            st.session_state.messages.append(
                {"role": "assistant", "content": "I Don't know"}
            )
            st.chat_message("assistant").write("I Don't know")
        else:
            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type_kwargs={"prompt": QA_PROMPT},
                return_source_documents=True,
            )
            result = rag_chain({"query": prompt})
            response = result["result"]
            sources = result["source_documents"]

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

            if sources:
                with st.expander("Source Documents"):
                    for doc in sources:
                        st.write(f"- {doc.metadata['source']}")
    else:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Chatbot is not ready (check API key or local LLM loading).",
            }
        )
        st.chat_message("assistant").write(
            "Chatbot is not ready (check API key or local LLM loading)."
        )
