# 📄 RAG Chatbot — Chat with Your PDFs

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload PDF documents and ask questions about them. Supports **4 LLM providers** — including 2 completely free options.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 What It Does

1. **Upload** any PDF (contracts, manuals, reports, research papers)
2. **Automatic chunking** — splits documents into semantic chunks and stores embeddings in ChromaDB
3. **Ask questions** — retrieves relevant chunks and generates answers with page citations
4. **4 LLM providers** — swap between local and cloud models with one click

## 🤖 Supported Providers

| Provider | Cost | Setup |
|----------|------|-------|
| **Ollama** | 🟢 Free (local) | [Install Ollama](https://ollama.com) → `ollama pull llama3.1` |
| **Groq** | 🟢 Free tier | Get key at [console.groq.com](https://console.groq.com) |
| **OpenAI** | 🔴 Paid | API key from [platform.openai.com](https://platform.openai.com) |
| **Anthropic** | 🔴 Paid | API key from [console.anthropic.com](https://console.anthropic.com) |

## 🏗️ Architecture

```
PDF Upload → PyPDF Loader → Text Splitter (800 char chunks)
                                    ↓
                            ChromaDB (vector store)
                                    ↓
User Query → Cosine Similarity Search → Top-K chunks → LLM → Answer + Citations
```

## ⚡ Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env — for Ollama you don't need any keys!

# (If using Ollama) Pull a model
ollama pull llama3.1

# Run the app
streamlit run app.py
```

## 📁 Project Structure

```
rag-chatbot/
├── app.py                 # Streamlit UI (provider selector, chat, upload)
├── rag/
│   ├── __init__.py
│   └── engine.py          # RAG pipeline: ingest → retrieve → generate
├── chroma_db/             # Vector store (auto-created, gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

## 🧠 Technical Decisions

- **ChromaDB** — zero-config vector store, runs locally, no external services needed
- **Multi-provider architecture** — abstract LLM layer lets clients use their preferred provider
- **RecursiveCharacterTextSplitter** — 800-char chunks with 100-char overlap balances context and precision
- **Cosine similarity** — standard metric for semantic text search
- **Temperature 0.2** — low creativity keeps answers grounded in source documents

## 📸 Demo

> 🎥 [Watch the 2-min demo on Loom](#) *(add your Loom link here)*

## 🚀 Potential Improvements

- [ ] Support for DOCX, TXT, and CSV files
- [ ] Conversation memory (multi-turn context)
- [ ] Streaming responses
- [ ] Authentication & multi-user support
- [ ] Deploy to AWS/GCP with Docker

## 📝 License

MIT
