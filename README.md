# Local Customer Support Chatbot

This project implements a local Retrieval-Augmented Generation (RAG) chatbot trained on customer support documentation (PDF and DOCX files) to answer user queries. The chatbot uses Sentence Transformers for embedding the documents and a locally downloaded Large Language Model (LLM) for generating answers.

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd rag_chatbot
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install transformers torch
    ```
    The `requirements.txt` file specifies the following library versions for reproducibility (as of April 14, 2025):
    ```
    chromadb==0.6.3
    pdfminer.six==20250327
    streamlit==1.44.1
    python-dotenv==1.1.0
    openai==1.73.0
    tiktoken==0.9.0
    python-docx==1.1.2
    sentence-transformers==4.0.2
    langchain-community==0.3.21
    langchain-chroma==0.2.2
    langchain-huggingface==0.1.2
    langchain-openai==0.3.12
    watchdog==6.0.0
    ```
    You can install these specific versions using the command above, along with `transformers` and `torch` for the local LLM.

3.  **Place Documentation:**
    * Place your customer support documentation files (both `.pdf` and `.docx`) in the `data/support_documentation` directory. Text files (`.txt`) are also supported.

4.  **Run Data Ingestion:**
    ```bash
    python chatbot/ingest.py
    ```
    This script will process your PDF, DOCX, and TXT documents, create embeddings using Sentence Transformers, and store them in a ChromaDB vector database (`chroma_db` directory).

5.  **Run the Chatbot Interface (Locally):**
    ```bash
    streamlit run chatbot/chatbot_interface.py
    ```
    This will open the chatbot interface in your web browser. The chatbot will download and load a local LLM (currently configured as `TinyLlama/TinyLlama-1.1B-Chat-v1.0`).

## Usage

1.  Open the web interface (either the local Streamlit app).
2.  Type your questions related to the customer support documentation in the chat input field.
3.  The chatbot will retrieve relevant information from the documentation and use the local LLM to generate an answer.
4.  Source documents used for retrieval are also displayed for context.

## Notes and Limitations

* The chatbot operates entirely locally. No external API keys (like OpenAI) are required for basic functionality.
* The quality of the answers depends on the relevance of the retrieved documents and the capabilities of the chosen local LLM (`TinyLlama/TinyLlama-1.1B-Chat-v1.0` by default). Larger local LLMs might provide better answers but require more computational resources.
* **Hardware Requirements:** Running local LLMs can be resource-intensive. Ensure your system has sufficient RAM (ideally 8GB or more) and consider using a machine with a dedicated GPU for better performance.
* The first run might take longer as the local LLM needs to be downloaded from Hugging Face.
* You can experiment with different local LLM models by changing the `LOCAL_LLM_MODEL_NAME` variable in `chatbot/rag_pipeline.py` and `chatbot/chatbot_interface.py`. Refer to the Hugging Face model hub ([https://huggingface.co/models](https://huggingface.co/models)) for available options. Smaller, quantized models might be necessary for systems with limited resources.
* The accuracy and coherence of the generated answers from local LLMs can vary.

## Optional: Using OpenAI API (Requires Configuration)

The codebase still retains the functionality to use the OpenAI API for embeddings and language generation if desired. To enable this:

1.  **Create a `.env` file** in the root directory. A sample `.env` file would look like this:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    USE_OPENAI="True"
    CHROMA_PATH="chroma_db"
    LOCAL_LLM_MODEL_NAME_ENV="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # TOKENIZERS_PARALLELISM=false
    ```
    Replace `"your_openai_api_key_here"` with your actual OpenAI API key. Set `USE_OPENAI` to `"True"` to enable OpenAI. If this line is absent or set to `"False"`, the chatbot will operate in local-only mode.

2.  **Run `ingest.py` and `streamlit run chatbot/chatbot_interface.py` as before.** The chatbot will now use the OpenAI API if configured in `.env`.
