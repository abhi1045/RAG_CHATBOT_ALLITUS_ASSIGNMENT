import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self):
        self.chroma_path = os.getenv("CHROMA_PATH", "chroma_db")
        self.use_openai = os.getenv("USE_OPENAI", "False").lower() == "true"
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.local_llm_model_name = os.getenv(
            "LOCAL_LLM_MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
        self.data_path = "data/support_documentation"
        self.prompt_template = """Use the following pieces of context to answer the question at the end.
1. Your bot should be able to answer questions only based on information present in the documents.
2. It should strictly reply back with `I Don't know` if a question is asked from outside these information sources.
3. Build a user-friendly chatbot interface to demonstrate the chatbot.

{context}

Question: {question}
Helpful Answer:"""


settings = Settings()
