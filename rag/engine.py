"""
RAG Engine - Handles document ingestion, vector storage, and retrieval-augmented generation.

Supported LLM providers:
  - Ollama (FREE, local) — no API key needed, runs on your machine
  - Groq  (FREE tier)    — fast cloud inference, free API key at console.groq.com
  - OpenAI               — requires paid API key
  - Anthropic            — requires paid API key
"""

import os
from pathlib import Path

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "rag_documents"

# ── Provider configs ─────────────────────────────────────────
PROVIDERS = {
    "ollama": {
        "name": "Ollama (Local, Free)",
        "needs_key": False,
        "default_model": "llama3.1",
    },
    "groq": {
        "name": "Groq (Cloud, Free Tier)",
        "needs_key": True,
        "env_var": "GROQ_API_KEY",
        "default_model": "llama-3.1-8b-instant",
    },
    "openai": {
        "name": "OpenAI (Paid)",
        "needs_key": True,
        "env_var": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    "anthropic": {
        "name": "Anthropic Claude (Paid)",
        "needs_key": True,
        "env_var": "ANTHROPIC_API_KEY",
        "default_model": "claude-sonnet-4-20250514",
    },
}


# ── ChromaDB helpers ─────────────────────────────────────────
def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_or_create_collection(client: chromadb.PersistentClient):
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ── Document ingestion ──────────────────────────────────────
def ingest_pdf(file_path: str) -> dict:
    """Load a PDF, split into chunks, store in ChromaDB."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    client = get_chroma_client()
    collection = get_or_create_collection(client)
    file_name = Path(file_path).name

    ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]
    documents = [chunk.page_content for chunk in chunks]
    metadatas = [
        {
            "source": file_name,
            "page": chunk.metadata.get("page", 0),
            "chunk_index": i,
        }
        for i, chunk in enumerate(chunks)
    ]

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    return {"file": file_name, "pages": len(pages), "chunks": len(chunks)}


# ── Retrieval ────────────────────────────────────────────────
def retrieve_context(query: str, n_results: int = 5) -> list[dict]:
    """Search ChromaDB for the most relevant chunks."""
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    if collection.count() == 0:
        return []

    results = collection.query(query_texts=[query], n_results=n_results)

    return [
        {
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", "unknown"),
            "page": results["metadatas"][0][i].get("page", 0),
            "score": round(1 - results["distances"][0][i], 3),
        }
        for i in range(len(results["documents"][0]))
    ]


# ── Prompt builder ───────────────────────────────────────────
def build_prompt(query: str, context_items: list[dict]) -> str:
    context_block = "\n\n---\n\n".join(
        f"[Source: {item['source']}, Page {item['page'] + 1}]\n{item['text']}"
        for item in context_items
    )

    return f"""You are a helpful assistant that answers questions based on the provided documents.
Use ONLY the context below to answer. If the answer is not in the context, say so clearly.
Always cite which document and page number your answer comes from.

CONTEXT:
{context_block}

QUESTION: {query}

ANSWER:"""


# ── LLM providers ───────────────────────────────────────────
def chat_ollama(query: str, context_items: list[dict], model: str = "llama3.1") -> str:
    """Generate response using Ollama (local, free)."""
    import requests

    prompt = build_prompt(query, context_items)
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def chat_groq(query: str, context_items: list[dict], model: str = "llama-3.1-8b-instant") -> str:
    """Generate response using Groq (free tier, fast)."""
    from groq import Groq

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = build_prompt(query, context_items)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1000,
    )
    return response.choices[0].message.content


def chat_openai(query: str, context_items: list[dict], model: str = "gpt-4o-mini") -> str:
    """Generate response using OpenAI."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = build_prompt(query, context_items)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1000,
    )
    return response.choices[0].message.content


def chat_anthropic(query: str, context_items: list[dict], model: str = "claude-sonnet-4-20250514") -> str:
    """Generate response using Anthropic Claude."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    prompt = build_prompt(query, context_items)

    response = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


_CHAT_FNS = {
    "ollama": chat_ollama,
    "groq": chat_groq,
    "openai": chat_openai,
    "anthropic": chat_anthropic,
}


# ── Main RAG pipeline ───────────────────────────────────────
def ask(query: str, provider: str = "ollama", n_results: int = 5) -> dict:
    """Full RAG pipeline: retrieve context → generate answer."""
    context_items = retrieve_context(query, n_results=n_results)

    if not context_items:
        return {
            "answer": "No documents have been uploaded yet. Please upload a PDF first.",
            "sources": [],
        }

    chat_fn = _CHAT_FNS.get(provider)
    if not chat_fn:
        raise ValueError(f"Unknown provider: {provider}")

    answer = chat_fn(query, context_items)
    sources = list({f"{item['source']} (p. {item['page'] + 1})" for item in context_items})

    return {"answer": answer, "sources": sources}


# ── Collection management ───────────────────────────────────
def get_collection_stats() -> dict:
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    count = collection.count()

    if count == 0:
        return {"total_chunks": 0, "documents": []}

    all_meta = collection.get()["metadatas"]
    unique_sources = list({m["source"] for m in all_meta})
    return {"total_chunks": count, "documents": unique_sources}


def clear_collection():
    client = get_chroma_client()
    client.delete_collection(COLLECTION_NAME)
