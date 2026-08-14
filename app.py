import hashlib
from pathlib import Path

import streamlit as st

from rag_embedding import (
    read_pdf_pages,
    build_chunks,
    store_in_chroma
)

from rag_chat import (
    get_cohere_client,
    load_collection,
    retrieve_context,
    build_context,
    answer_question
)


BASE_DIR = Path(
    __file__
).resolve().parent

DB_PATH = (
    BASE_DIR /
    "chroma_db"
)

UPLOAD_DIR = (
    BASE_DIR /
    "uploads"
)

COLLECTION_NAME = (
    "startup_knowledge"
)


# --------------------------------
# Streamlit
# --------------------------------

st.set_page_config(

    page_title=
        "Startup AI Agent",

    page_icon=
        "🚀",

    layout=
        "centered"
)


st.title(
    "🚀 Startup AI Agent"
)

st.caption(
    "AI assistant for entrepreneurs "
    "to discover startup schemes, "
    "funding, investors and infrastructure."
)


# --------------------------------
# Session state
# --------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


if "indexed_files" not in st.session_state:

    st.session_state.indexed_files = {}


# --------------------------------
# Save PDF
# --------------------------------

def save_pdf(uploaded_file):

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    file_bytes = (
        uploaded_file
        .getvalue()
    )

    file_hash = hashlib.sha1(
        file_bytes
    ).hexdigest()

    path = (
        UPLOAD_DIR /
        f"{file_hash}-{uploaded_file.name}"
    )

    path.write_bytes(
        file_bytes
    )

    return path, file_hash


# --------------------------------
# Index PDFs
# --------------------------------

def index_pdfs(files):

    all_records = []

    added = []

    for uploaded_file in files:

        pdf_path, file_hash = (
            save_pdf(
                uploaded_file
            )
        )

        if (
            file_hash
            in st.session_state
            .indexed_files
        ):

            continue

        pages = read_pdf_pages(
            pdf_path
        )

        records = build_chunks(

            pages=pages,

            source_name=
                uploaded_file.name
        )

        all_records.extend(
            records
        )

        st.session_state\
            .indexed_files[
                file_hash
            ] = uploaded_file.name

        added.append(
            uploaded_file.name
        )


    total = store_in_chroma(

        records=all_records,

        db_path=DB_PATH,

        collection_name=
            COLLECTION_NAME
    )


    return added, len(
        all_records
    ), total


# --------------------------------
# Sidebar
# --------------------------------

with st.sidebar:

    st.header(
        "📚 Startup Knowledge Base"
    )

    uploaded_files = (
        st.file_uploader(

            "Upload startup PDFs",

            type=["pdf"],

            accept_multiple_files=True
        )
    )


    if st.button(
        "Index PDFs",
        use_container_width=True
    ):

        if not uploaded_files:

            st.warning(
                "Upload PDFs first."
            )

        else:

            with st.spinner(
                "Processing PDFs..."
            ):

                added, chunks, total = (
                    index_pdfs(
                        uploaded_files
                    )
                )


            st.success(
                f"Added {len(added)} "
                f"file(s), {chunks} chunks."
            )

            st.write(
                f"Total records: {total}"
            )


    st.subheader(
        "Indexed Documents"
    )


    if (
        st.session_state
        .indexed_files
    ):

        for filename in (
            st.session_state
            .indexed_files
            .values()
        ):

            st.write(
                f"✓ {filename}"
            )

    else:

        st.write(
            "No documents indexed."
        )


# --------------------------------
# Chat history
# --------------------------------

for message in (
    st.session_state
    .messages
):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["text"]
        )


# --------------------------------
# Chat
# --------------------------------

question = st.chat_input(

    "Ask about government schemes, "
    "investors, funding or infrastructure..."
)


if question:

    if not st.session_state.indexed_files:

        st.warning(
            "Please upload and index "
            "your startup PDFs first."
        )

        st.stop()


    st.session_state.messages.append({

        "role":
            "user",

        "text":
            question
    })


    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching knowledge base..."
        ):

            try:

                collection = (
                    load_collection(
                        DB_PATH
                    )
                )

                chunks = retrieve_context(

                    collection,

                    question,

                    top_k=5
                )


                context = (
                    build_context(
                        chunks
                    )
                )


                if not context:

                    answer = (
                        "I could not find "
                        "relevant information "
                        "in the knowledge base."
                    )

                else:

                    client = (
                        get_cohere_client()
                    )

                    answer = (
                        answer_question(

                            client,

                            question,

                            context
                        )
                    )


                st.markdown(
                    answer
                )


                with st.expander(
                    "Retrieved Sources"
                ):

                    for chunk in chunks:

                        metadata = (
                            chunk["metadata"]
                        )

                        st.write(
                            f"Source: "
                            f"{metadata.get('source')}"
                        )

                        st.write(
                            f"Page: "
                            f"{metadata.get('page')}"
                        )

                        st.caption(
                            chunk["document"][
                                :500
                            ]
                        )


                st.session_state\
                    .messages.append({

                        "role":
                            "assistant",

                        "text":
                            answer
                    })


            except Exception as error:

                st.error(
                    f"Error: {error}"
                )