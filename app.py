import hashlib
from pathlib import Path

import streamlit as st

from voice import speech_to_text

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


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "chroma_db"

UPLOAD_DIR = BASE_DIR / "uploads"

COLLECTION_NAME = "startup_knowledge"


# ==========================================
# STREAMLIT CONFIG
# ==========================================

st.set_page_config(
    page_title="Startup AI Agent",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        * {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
        }

        .stApp {
            background: linear-gradient(135deg, #f8fafb 0%, #f3f4f6 50%, #ecf0f5 100%);
        }

        .block-container {
            max-width: 1400px;
            padding-top: 2rem;
            padding-bottom: 3rem;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
        }

        /* SIDEBAR STYLING */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.95) 100%);
            border-right: 1px solid rgba(209, 213, 219, 0.4);
            backdrop-filter: blur(10px);
        }

        section[data-testid="stSidebar"] h2 {
            font-size: 1.1rem;
            font-weight: 700;
            color: #1f2937;
            margin-top: 1.2rem !important;
            margin-bottom: 0.8rem !important;
            letter-spacing: -0.01em;
        }

        section[data-testid="stSidebar"] h3 {
            font-size: 0.95rem;
            font-weight: 600;
            color: #374151;
            margin-top: 0.8rem !important;
        }

        /* HERO SECTION */
        .hero-shell {
            background: linear-gradient(135deg, rgba(255,255,255,0.6) 0%, rgba(240,244,250,0.8) 50%, rgba(235,245,250,0.7) 100%);
            border: 1.5px solid rgba(199, 210, 254, 0.3);
            border-radius: 24px;
            padding: 2.5rem 3rem 2rem 3rem;
            box-shadow: 0 20px 45px rgba(79, 70, 229, 0.08), inset 0 1px 0 rgba(255,255,255,0.8);
            margin: 0 auto 2.5rem auto;
            max-width: 1200px;
        }

        .rocket-icon {
            display: inline-flex;
            width: 56px;
            height: 56px;
            align-items: center;
            justify-content: center;
            font-size: 2.8rem;
            margin-right: 1rem;
            background: linear-gradient(135deg, rgba(79, 70, 229, 0.1), rgba(59, 130, 246, 0.1));
            border-radius: 16px;
            filter: drop-shadow(0 10px 15px rgba(79, 70, 229, 0.15));
        }

        .hero-title {
            font-size: clamp(2.2rem, 3.5vw, 3.5rem);
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #0f172a;
            margin: 0;
            line-height: 1.2;
        }

        .hero-subtitle {
            font-size: 1rem;
            color: #475569;
            margin-top: 1.2rem;
            max-width: 900px;
            line-height: 1.8;
            font-weight: 400;
        }

        .badge {
            display: inline-block;
            margin-top: 1.2rem;
            background: linear-gradient(135deg, #dbeafe 0%, #e0e7ff 100%);
            border: 1px solid rgba(79, 70, 229, 0.2);
            border-radius: 20px;
            padding: 0.5rem 1.2rem;
            color: #3730a3;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        /* INPUT SECTIONS */
        .input-section {
            max-width: 1000px;
            margin: 2rem auto 0 auto;
            padding: 2rem 2.5rem;
            background: rgba(255, 255, 255, 0.6);
            border: 1px solid rgba(209, 213, 219, 0.3);
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
        }

        .section-header {
            font-size: 1.05rem;
            font-weight: 700;
            color: #1f2937;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin: 0 0 1rem 0;
            letter-spacing: -0.01em;
        }

        .section-header .emoji {
            font-size: 1.5rem;
            opacity: 1;
        }

        /* FORM INPUTS */
        .stTextInput > div > div > input,
        .stChatInput textarea,
        .stTextArea textarea {
            background: rgba(255,255,255,0.85) !important;
            color: #111827 !important;
            border: 1.5px solid rgba(209, 213, 219, 0.5) !important;
            border-radius: 12px !important;
            padding: 0.75rem 1rem !important;
            font-size: 0.95rem !important;
            font-weight: 500 !important;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
            transition: all 0.2s ease !important;
        }

        .stTextInput > div > div > input:focus,
        .stChatInput textarea:focus,
        .stTextArea textarea:focus {
            border-color: rgba(79, 70, 229, 0.6) !important;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.15) !important;
        }

        .stChatInput {
            background: rgba(255,255,255,0.7) !important;
            border: 1.5px solid rgba(209, 213, 219, 0.3) !important;
            border-radius: 16px !important;
            padding: 0.5rem 0.5rem !important;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06) !important;
        }

        .stChatInput button {
            border-radius: 10px !important;
            background: linear-gradient(135deg, #4f46e5, #3b82f6) !important;
            border: none !important;
            color: white !important;
            font-weight: 700 !important;
            padding: 0.6rem 1.2rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
        }

        .stChatInput button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4) !important;
        }

        /* BUTTONS */
        .stButton > button {
            border-radius: 12px;
            border: 1.5px solid rgba(209, 213, 219, 0.5);
            background: rgba(255,255,255,0.7);
            color: #1f2937;
            font-weight: 700;
            padding: 0.6rem 1.2rem;
            font-size: 0.95rem;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
        }

        .stButton > button:hover {
            background: rgba(255,255,255,0.9);
            border-color: rgba(79, 70, 229, 0.3);
            color: #4f46e5;
            box-shadow: 0 4px 16px rgba(79, 70, 229, 0.15);
            transform: translateY(-1px);
        }

        /* ALERTS AND MESSAGES */
        .stAlert {
            border-radius: 12px;
            border: 1.5px solid;
            padding: 1rem 1.25rem;
            font-weight: 500;
        }

        .stSuccess {
            background: rgba(220, 252, 231, 0.7) !important;
            border-color: rgba(34, 197, 94, 0.3) !important;
            color: #065f46 !important;
        }

        .stError {
            background: rgba(254, 226, 226, 0.7) !important;
            border-color: rgba(239, 68, 68, 0.3) !important;
            color: #7f1d1d !important;
        }

        .stWarning {
            background: rgba(254, 243, 199, 0.7) !important;
            border-color: rgba(202, 138, 4, 0.3) !important;
            color: #78350f !important;
        }

        .stInfo {
            background: rgba(219, 234, 254, 0.7) !important;
            border-color: rgba(59, 130, 246, 0.3) !important;
            color: #0c2340 !important;
        }

        /* CHAT MESSAGES */
        .stChatMessage {
            background: rgba(255, 255, 255, 0.5) !important;
            border: 1px solid rgba(209, 213, 219, 0.2) !important;
            border-radius: 16px !important;
            padding: 1.25rem !important;
            margin-bottom: 1rem !important;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04) !important;
        }

        /* EXPANDER */
        .streamlit-expanderHeader {
            background: rgba(248, 250, 252, 0.8) !important;
            border: 1px solid rgba(209, 213, 219, 0.3) !important;
            border-radius: 12px !important;
        }

        /* AUDIO INPUT */
        .stAudioInput {
            border-radius: 12px;
            border: 1.5px solid rgba(209, 213, 219, 0.3);
            background: rgba(255,255,255,0.7);
            padding: 0.75rem;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }

        /* DIVIDER */
        hr {
            border-color: rgba(209, 213, 219, 0.3) !important;
            margin: 1.5rem 0 !important;
        }

        /* TEXT STYLING */
        p, span, label {
            color: #374151;
            font-weight: 500;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #0f172a;
            font-weight: 700;
            letter-spacing: -0.01em;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-shell">
        <div style="display:flex; align-items:center; gap:12px;">
            <div class="rocket-icon">🚀</div>
            <h1 class="hero-title">Startup AI Agent</h1>
        </div>
        <div class="hero-subtitle">
            Your AI assistant for entrepreneurs to discover government schemes,
            funding, investors and startup infrastructure.
        </div>
        <div class="badge">24/7 guidance</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = {}


# ==========================================
# SAVE PDF
# ==========================================

def save_pdf(uploaded_file):

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    file_bytes = uploaded_file.getvalue()

    file_hash = hashlib.sha1(
        file_bytes
    ).hexdigest()

    file_path = (
        UPLOAD_DIR /
        f"{file_hash}-{uploaded_file.name}"
    )

    file_path.write_bytes(file_bytes)

    return file_path, file_hash


# ==========================================
# INDEX PDF FILES
# ==========================================

def index_pdfs(files):

    all_records = []

    added_files = []

    for uploaded_file in files:

        pdf_path, file_hash = save_pdf(
            uploaded_file
        )

        # Avoid duplicate indexing
        if file_hash in st.session_state.indexed_files:
            continue

        # Read PDF
        pages = read_pdf_pages(
            pdf_path
        )

        # Create chunks
        records = build_chunks(
            pages=pages,
            source_name=uploaded_file.name
        )

        all_records.extend(records)

        st.session_state.indexed_files[
            file_hash
        ] = uploaded_file.name

        added_files.append(
            uploaded_file.name
        )

    # Store in ChromaDB
    total_records = store_in_chroma(
        records=all_records,
        db_path=DB_PATH,
        collection_name=COLLECTION_NAME
    )

    return (
        added_files,
        len(all_records),
        total_records
    )


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding: 1rem 0 1.5rem 0;">
            <h2 style="margin: 0; font-size: 1.1rem; font-weight: 800; color: #0f172a;">
                📚 Knowledge Base
            </h2>
            <p style="font-size: 0.85rem; color: #6b7280; margin: 0.5rem 0 0;">
                Upload and index startup documents
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "📄 Select PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([2, 1])
    
    with col1:
        index_btn = st.button(
            "📥 Index PDFs",
            use_container_width=True
        )
    
    with col2:
        clear_btn = st.button(
            "🗑️",
            use_container_width=True,
            help="Clear all documents"
        )

    if index_btn:
        if not uploaded_files:
            st.warning("📌 Please select PDF files first.")
        else:
            with st.spinner("⏳ Processing PDFs..."):
                added, chunks, total = index_pdfs(uploaded_files)
            st.success(
                f"✅ Added **{len(added)}** file(s) with **{chunks}** chunks"
            )
            st.info(f"📊 Total records: **{total}**")

    st.divider()

    st.markdown(
        """
        <div style="padding: 0.5rem 0;">
            <h3 style="margin: 0; font-size: 0.95rem; font-weight: 700; color: #0f172a;">
                ✅ Indexed Documents
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.indexed_files:
        for filename in st.session_state.indexed_files.values():
            st.markdown(
                f"""
                <div style="
                    background: rgba(220, 252, 231, 0.5);
                    border: 1px solid rgba(34, 197, 94, 0.2);
                    border-radius: 8px;
                    padding: 0.6rem 0.8rem;
                    margin-bottom: 0.5rem;
                    font-size: 0.85rem;
                    color: #065f46;
                    font-weight: 600;
                ">
                ✅ {filename}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div style="
                text-align: center;
                padding: 1.5rem 0.5rem;
                color: #9ca3af;
                font-size: 0.85rem;
            ">
            📋 No documents indexed yet
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    if clear_btn:
        st.session_state.messages = []
        st.session_state.indexed_files = {}
        st.rerun()

    st.markdown(
        """
        <div style="
            padding: 1.5rem 0.75rem;
            background: linear-gradient(135deg, rgba(79, 70, 229, 0.08), rgba(59, 130, 246, 0.08));
            border: 1px solid rgba(79, 70, 229, 0.1);
            border-radius: 12px;
            text-align: center;
            font-size: 0.8rem;
            color: #4f46e5;
            font-weight: 600;
            margin-top: 1rem;
        ">
        💡 Tip: Upload PDFs to get AI-powered answers
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================

if st.session_state.messages:
    st.markdown("---")
    st.markdown(
        """
        <div style="margin: 1.5rem 0;">
            <h3 style="font-size: 0.95rem; font-weight: 700; color: #1f2937; text-transform: uppercase; letter-spacing: 0.05em;">
                💬 Conversation History
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍💼" if message["role"] == "user" else "🤖"):
        st.markdown(message["text"])


# ==========================================
# INPUT SECTION
# ==========================================

st.markdown(
    """
    <div class="input-section">
        <div style="display: flex; gap: 2rem; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 300px;">
                <div class="section-header"><span class="emoji">⌨️</span> Ask your question</div>
                <p style="font-size: 0.85rem; color: #6b7280; margin: 0 0 0.75rem 0;">
                    Type to ask about government schemes, funding, investors or startup resources.
                </p>
            </div>
            <div style="flex: 1; min-width: 300px;">
                <div class="section-header"><span class="emoji">🎤</span> Or use voice</div>
                <p style="font-size: 0.85rem; color: #6b7280; margin: 0 0 0.75rem 0;">
                    Click the microphone icon and speak your question naturally.
                </p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1.2, 1])

with col1:
    text_question = st.chat_input(
        "Ask about schemes, investors, funding or infrastructure...",
        key="text_input"
    )

with col2:
    audio = st.audio_input(
        "Click here and speak",
        sample_rate=16000,
        key="voice_input"
    )


voice_question = ""

if audio is not None:
    with st.spinner("🎧 Converting speech to text..."):
        voice_question = speech_to_text(audio)

    if voice_question:
        st.success(
            f"✅ **You said:** _{voice_question}_"
        )
    else:
        st.error(
            "😕 Sorry, I couldn't understand the audio. Please try again."
        )


# ==========================================
# SELECT USER INPUT
# ==========================================

question = None

if text_question:

    question = text_question

elif voice_question:

    question = voice_question


# ==========================================
# PROCESS QUESTION
# ==========================================

if question:

    # Check whether PDFs are indexed
    if not st.session_state.indexed_files:
        st.error(
            "📋 **Please upload and index startup PDFs first** to enable AI-powered answers."
        )
        st.stop()

    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "text": question
    })

    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant", avatar="🤖"):
        try:
            # Search ChromaDB
            with st.spinner("🔍 Searching knowledge base..."):
                collection = load_collection(DB_PATH)
                chunks = retrieve_context(collection, question, top_k=5)
                context = build_context(chunks)

            # Generate answer
            if not context:
                answer = (
                    "❌ I couldn't find relevant information in the current knowledge base. "
                    "Please upload more documents related to your question."
                )
            else:
                with st.spinner("🤖 Generating answer with AI..."):
                    client = get_cohere_client()
                    answer = answer_question(client, question, context)

            # Display answer
            st.markdown(answer)

            # Retrieved sources
            if chunks:
                with st.expander("📚 Retrieved Sources", expanded=False):
                    st.markdown(
                        """
                        <p style="font-size: 0.85rem; color: #6b7280; margin-bottom: 1rem;">
                        The following documents were used to generate this answer:
                        </p>
                        """,
                        unsafe_allow_html=True,
                    )
                    for index, chunk in enumerate(chunks, start=1):
                        metadata = chunk["metadata"]
                        source = metadata.get("source", "Unknown")
                        page = metadata.get("page", "?")

                        st.markdown(
                            f"""
                            <div style="
                                background: rgba(248, 250, 252, 0.8);
                                border-left: 4px solid #4f46e5;
                                border-radius: 8px;
                                padding: 1rem;
                                margin-bottom: 0.75rem;
                            ">
                                <p style="margin: 0 0 0.5rem 0; font-weight: 700; color: #1f2937;">
                                {index}. {source} <span style="color: #9ca3af; font-size: 0.85rem;">(Page {page})</span>
                                </p>
                                <p style="margin: 0; font-size: 0.85rem; color: #6b7280; line-height: 1.6;">
                                {chunk['document'][:400]}...
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # Save assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "text": answer
            })

        except Exception as error:
            st.error(
                f"❌ **An error occurred:** {error}\n\n"
                f"Please try again or check your settings."
            )