"""
RAG Chatbot - Streamlit UI
Upload PDFs and chat with your documents using local or cloud LLMs.
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from rag.engine import ask, ingest_pdf, get_collection_stats, clear_collection, PROVIDERS

load_dotenv()

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")

st.title("📄 RAG Chatbot")
st.caption("Upload PDFs → Ask questions → Get answers with citations")

# ── Sidebar: Config & Upload ────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    provider = st.selectbox(
        "LLM Provider",
        options=list(PROVIDERS.keys()),
        format_func=lambda x: PROVIDERS[x]["name"],
    )

    # Validate provider setup
    prov_config = PROVIDERS[provider]
    if prov_config["needs_key"]:
        env_var = prov_config["env_var"]
        if not os.getenv(env_var):
            st.warning(f"Set {env_var} in your .env file")
    elif provider == "ollama":
        st.info("Make sure Ollama is running: `ollama serve`")

    n_results = st.slider("Context chunks to retrieve", 3, 10, 5)

    st.divider()

    # File upload
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            with st.spinner(f"Processing {uploaded_file.name}..."):
                stats = ingest_pdf(tmp_path)
                st.success(
                    f"✅ **{stats['file']}** — {stats['pages']} pages, {stats['chunks']} chunks"
                )

            os.unlink(tmp_path)

    st.divider()

    # Collection info
    st.header("📊 Knowledge Base")
    col_stats = get_collection_stats()
    st.metric("Total chunks", col_stats["total_chunks"])

    if col_stats["documents"]:
        st.write("**Indexed documents:**")
        for doc in col_stats["documents"]:
            st.write(f"- {doc}")

    if st.button("🗑️ Clear all documents", type="secondary"):
        clear_collection()
        st.rerun()

# ── Chat Interface ──────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents & generating answer..."):
            result = ask(prompt, provider=provider, n_results=n_results)

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander("📎 Sources"):
                for source in result["sources"]:
                    st.write(f"- {source}")

    full_response = result["answer"]
    if result["sources"]:
        full_response += "\n\n**Sources:** " + ", ".join(result["sources"])
    st.session_state.messages.append({"role": "assistant", "content": full_response})
