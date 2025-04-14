# Customer Support Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot trained on customer support documentation (PDF and DOCX files) to answer user queries.

## Setup Instructions

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd rag_chatbot
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` file specifies the following library versions for reproducibility (as of April 14, 2025):

    ```
    langchain==0.1.17
    chromadb==1.0.4
    pdfminer.six==20250327
    streamlit==1.34.0
    python-dotenv==1.0.1
    openai==1.73.0
    tiktoken==0.9.0
    python-docx==1.1.1
    sentence-transformers==4.0.2
    ```

    You can install these specific versions using the command above.

3.  **Set up OpenAI API Key:**

    - Create a `.env` file in the root directory.
    - Add your OpenAI API key to the `.env` file:
      ```
      OPENAI_API_KEY="your_openai_api_key_here"
      ```

4.  **Place Documentation:**

    - Place your customer support documentation files (both `.pdf` and `.docx`) in the `data/support_documentation` directory. Text files (`.txt`) are also supported.

5.  **Run Data Ingestion:**

    ```bash
    python chatbot/ingest.py
    ```

    This script will process your PDF, DOCX, and TXT documents, create embeddings, and store them in a ChromaDB vector database (`chroma_db` directory).

6.  **Run the Chatbot Interface (Locally):**
    ```bash
    streamlit run chatbot/chatbot_interface.py
    ```
    This will open the chatbot interface in your web browser.

## Usage

1.  Open the web interface (either the local Streamlit app or the deployed version).
2.  Type your questions related to the customer support documentation in the chat input field.
3.  The chatbot will provide answers based on the information found in the PDF, DOCX, and TXT documents.
4.  If a question is outside the scope of the documentation, the chatbot should respond with "I Don't know."

## Notes and Limitations

- The accuracy of the answers depends on the quality and relevance of the provided customer support documentation.
- The chatbot uses OpenAI's language models, which may have their own limitations.
- The "I Don't know" functionality relies on the RAG pipeline's ability to not find relevant information and the language model's understanding to not hallucinate answers.
- This version of the chatbot supports ingesting and querying information from PDF (`.pdf`), DOCX (`.docx`), and plain text (`.txt`) files. It uses ChromaDB for the vector store instead of FAISS.
- The specific library versions in `requirements.txt` were chosen for stability as of April 14, 2025. Newer versions might be available, but compatibility is not guaranteed.
