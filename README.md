# 🚀 Startup AI Agent

A Streamlit-based AI assistant for startup founders and entrepreneurs that answers questions using a PDF knowledge base and Cohere's language model.

## Features

- Government scheme and policy lookup
- Funding and investor discovery
- Startup infrastructure and incubation info
- Registration and compliance guidance
- PDF ingestion and vector search with ChromaDB
- Voice input support via speech-to-text
- Streamlit chat interface

## Architecture

1. Upload startup-related PDFs
2. Extract text and split into chunks
3. Embed and store chunks in ChromaDB
4. Retrieve relevant context for a user question
5. Send context and question to Cohere for an answer

## Tech Stack

- Python 3.11+
- Streamlit
- ChromaDB
- Cohere
- PyPDF
- SpeechRecognition
- python-dotenv

## Project Structure

- `app.py` — Streamlit frontend and question processing
- `rag_embedding.py` — PDF loading, chunking, indexing
- `rag_chat.py` — retrieval and LLM answer generation
- `voice.py` — audio-to-text conversion
- `data/` — source PDF documents
- `uploads/` — saved uploaded files
- `chroma_db/` — vector database storage

## Setup

```bash
git clone https://github.com/your-username/startup-ai-agent.git
cd startup-ai-agent
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root with your Cohere API key:

```env
COHERE_API_KEY=your_api_key_here
```

You can copy the sample file:

```bash
copy .env.example .env
```

## Run the App

```bash
streamlit run app.py
```

## Notes

- The app expects startup-related PDFs to be uploaded and indexed before asking questions.
- If no relevant context is found, the app will respond that the current knowledge base does not contain enough information.
