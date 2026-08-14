from pathlib import Path
from uuid import uuid4

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader


def read_pdf_pages(pdf_path: Path):

    reader = PdfReader(str(pdf_path))

    pages = []

    for index, page in enumerate(
        reader.pages,
        start=1
    ):

        text = (
            page.extract_text() or ""
        ).strip()

        if text:

            pages.append({
                "page": index,
                "text": text
            })

    return pages


def chunk_text(
    text,
    chunk_size=900,
    chunk_overlap=150
):

    chunks = []

    start = 0

    step = (
        chunk_size -
        chunk_overlap
    )

    while start < len(text):

        end = start + chunk_size

        chunk = text[
            start:end
        ].strip()

        if chunk:

            chunks.append(chunk)

        start += step

    return chunks


def build_chunks(
    pages,
    source_name
):

    records = []

    for page_info in pages:

        page_number = page_info["page"]

        page_text = page_info["text"]

        chunks = chunk_text(
            page_text
        )

        for index, chunk in enumerate(
            chunks
        ):

            records.append({

                "id":
                    f"{source_name}-"
                    f"p{page_number}-"
                    f"c{index}-"
                    f"{uuid4().hex[:8]}",

                "document": chunk,

                "metadata": {

                    "source":
                        source_name,

                    "page":
                        page_number,

                    "chunk_index":
                        index
                }
            })

    return records


def store_in_chroma(
    records,
    db_path,
    collection_name="startup_knowledge"
):

    db_path.mkdir(
        parents=True,
        exist_ok=True
    )

    client = chromadb.PersistentClient(
        path=str(db_path)
    )

    embedding_function = (
        embedding_functions
        .DefaultEmbeddingFunction()
    )

    collection = (
        client
        .get_or_create_collection(
            name=collection_name,
            embedding_function=
                embedding_function
        )
    )

    if not records:

        return collection.count()

    ids = [
        item["id"]
        for item in records
    ]

    documents = [
        item["document"]
        for item in records
    ]

    metadatas = [
        item["metadata"]
        for item in records
    ]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    return collection.count()