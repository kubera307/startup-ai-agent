import os
from pathlib import Path

import chromadb
import cohere
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()


def get_cohere_client():
    """Return a configured Cohere client if an API key is available."""
    api_key = os.getenv("COHERE_API_KEY") or os.getenv("COHER_API_KEY")

    if not api_key:
        return None

    try:
        return cohere.ClientV2(api_key=api_key)
    except TypeError:
        return cohere.Client(api_key=api_key)


def load_collection(db_path, collection_name="startup_knowledge"):
    """Load or create the Chroma collection used for startup knowledge."""
    db_path = Path(db_path)
    db_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_path))
    embedding_function = embedding_functions.DefaultEmbeddingFunction()

    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
    )


def retrieve_context(collection, question, top_k=5):
    """Fetch the most relevant chunks from the collection."""
    if collection is None:
        return []

    try:
        result = collection.query(
            query_texts=[question],
            n_results=top_k,
        )
    except Exception:
        return []

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    chunks = []
    for doc, meta in zip(documents, metadatas):
        chunks.append({
            "document": doc,
            "metadata": meta or {},
        })

    return chunks


def build_context(chunks):
    """Combine retrieved chunks into a clean context string."""
    if not chunks:
        return ""

    return "\n\n".join(
        chunk.get("document", "")
        for chunk in chunks
        if chunk.get("document")
    )


def answer_question(client, question, context):
    """Generate an answer from Cohere using the retrieved context."""
    if client is None:
        return (
            "Please add a valid COHERE_API_KEY to your environment "
            "to enable AI responses."
        )

    prompt = (
        "Use the context below to answer the question accurately. "
        "If the answer is not in the context, say so clearly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    try:
        response = client.chat(
            model="command-r",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        if hasattr(response, "message") and response.message:
            message = response.message
            content = getattr(message, "content", None)

            if isinstance(content, list):
                for item in content:
                    if hasattr(item, "text"):
                        return item.text
                    if isinstance(item, dict) and "text" in item:
                        return item["text"]

            if isinstance(content, str):
                return content

        if hasattr(response, "text"):
            return response.text

        return str(response)

    except Exception as exc:
        return (
            f"I couldn't generate an answer right now: {str(exc)[:100]}\n\n"
            "Please check your Cohere API key or try again later."
        )
