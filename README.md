\# AI Knowledge Assistant – Local RAG System (Ollama + LangChain)



This project implements a \*\*100% local, cost-free Retrieval-Augmented Generation (RAG) system\*\* that allows users to query their own documents (PDFs) and receive \*\*grounded, source-backed answers\*\*.



It is designed to reflect \*\*enterprise-grade AI agent architectures\*\*, with a focus on reliability, transparency, and data privacy.



---



\## 🚀 Key Features



\- 🔒 \*\*Fully local inference\*\* (no external APIs, no data leakage)

\- 🤖 LLM-powered question answering using \*\*Ollama\*\*

\- 📚 Retrieval-Augmented Generation (RAG) with \*\*source attribution\*\*

\- 🧠 Semantic search with \*\*ChromaDB\*\*

\- 📄 Multi-PDF ingestion pipeline

\- 🌐 REST API exposed via \*\*FastAPI\*\*

\- 🧪 Interactive CLI + API usage

\- 🏗️ Clean, modular, production-oriented codebase



---



\## 🧠 Architecture Overview



```text

PDF Documents

&nbsp;    ↓

Text Splitting (LangChain)

&nbsp;    ↓

Embeddings (Hugging Face)

&nbsp;    ↓

Vector Store (ChromaDB)

&nbsp;    ↓

Retriever

&nbsp;    ↓

LLM (Ollama – Mistral / LLaMA)

&nbsp;    ↓

Answer + Sources



