import os
from pathlib import Path

import chromadb
import cohere

from chromadb.utils import embedding_functions

from dotenv import load_dotenv


load_dotenv()


MODEL = "command-a-plus-05-2026"

COLLECTION_NAME = "startup_knowledge"


def get_cohere_client():

    api_key = os.getenv(
        "COHERE_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "COHERE_API_KEY not found "
            "in .env"
        )

    return cohere.ClientV2(
        api_key=api_key
    )


def get_answer_text(response):

    content_items = (
        getattr(
            response.message,
            "content",
            []
        ) or []
    )

    text_parts = []

    for item in content_items:

        if (
            getattr(
                item,
                "type",
                None
            ) == "text"
            and
            hasattr(item, "text")
        ):

            text_parts.append(
                item.text
            )

    if text_parts:

        return "\n".join(
            text_parts
        ).strip()

    return (
        "No answer returned "
        "by the model."
    )


def load_collection(db_path):

    client = (
        chromadb
        .PersistentClient(
            path=str(db_path)
        )
    )

    embedding_function = (
        embedding_functions
        .DefaultEmbeddingFunction()
    )

    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=
            embedding_function
    )


def retrieve_context(
    collection,
    question,
    top_k=5
):

    result = collection.query(

        query_texts=[
            question
        ],

        n_results=top_k
    )

    documents = result.get(
        "documents",
        [[]]
    )

    metadatas = result.get(
        "metadatas",
        [[]]
    )

    distances = result.get(
        "distances",
        [[]]
    )

    rows = []

    for document, metadata, distance in zip(

        documents[0],

        metadatas[0],

        distances[0]
    ):

        rows.append({

            "document":
                document,

            "metadata":
                metadata or {},

            "distance":
                distance
        })

    return rows


def build_context(chunks):

    if not chunks:

        return ""

    lines = []

    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        metadata = (
            chunk["metadata"]
        )

        source = metadata.get(
            "source",
            "unknown"
        )

        page = metadata.get(
            "page",
            "?"
        )

        lines.append(
            f"Source {index}: "
            f"{source}, page {page}"
        )

        lines.append(
            chunk["document"]
        )

        lines.append("")

    return "\n".join(
        lines
    ).strip()


def answer_question(
    client,
    question,
    context
):

    system_prompt = """
You are Startup AI Agent.

Your purpose is to help entrepreneurs
set up and grow startups.

You provide information about:

- Government schemes
- Government funding
- Investors
- Funding options
- Startup infrastructure
- Incubators
- Accelerators
- Startup policies
- Startup registration
- Business planning

IMPORTANT RULES:

1. Answer using the provided
   knowledge base.

2. Do not invent government schemes,
   investor information, funding amounts,
   eligibility requirements or policies.

3. If the information is not available
   in the knowledge base, clearly say so.

4. For important government,
   financial or legal information,
   advise the entrepreneur to verify
   the latest details from the official
   source.

5. Give clear and practical answers.
"""


    user_prompt = f"""
Entrepreneur's question:

{question}


Knowledge base:

{context}


Answer the entrepreneur's question
using the knowledge base.
"""


    response = client.chat(

        model=MODEL,

        temperature=0,

        messages=[

            {
                "role": "system",

                "content":
                    system_prompt
            },

            {
                "role": "user",

                "content": [
                    {
                        "type": "text",

                        "text":
                            user_prompt
                    }
                ]
            }

        ]
    )


    return get_answer_text(
        response
    )