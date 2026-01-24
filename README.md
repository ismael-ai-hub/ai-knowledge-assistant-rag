## AI Knowledge Assistant with RAG (Local & Free)

This project implements a Retrieval-Augmented Generation (RAG) system that allows users to query their own documents (PDF files) and receive accurate, grounded, and context-aware answers.

The system is designed to be 100% local, privacy-friendly, and cost-free, relying exclusively on open-source tools widely used in modern AI and enterprise environments.

 ## Key Features

Local Large Language Model (LLM) inference using Ollama

Retrieval-Augmented Generation (RAG) for factual, source-grounded responses

Semantic vector search powered by ChromaDB

Open-source embeddings via Hugging Face sentence-transformers

PDF document ingestion and chunking

Persistent local vector storage

No external APIs, no API keys, no paid services

## Technology Stack
Component	Tool
LLM	Ollama (Mistral / LLaMA / Phi)
Embeddings	sentence-transformers (Hugging Face)
Vector Database	ChromaDB
RAG Framework	LangChain
Document Loader	PyPDF2
Language	Python
## Project Structure
ai-knowledge-assistant-rag/
│
├── data/                  # PDFs to ingest
│   └── .gitkeep
│
├── vectorstore/            # Persistent ChromaDB storage
│
├── rag_ingestion.py        # Document ingestion & embeddings
├── requirements.txt        # Python dependencies
└── README.md

## How It Works

PDFs are loaded from the data/ directory

Documents are split into overlapping text chunks

Embeddings are generated locally using Hugging Face models

Chunks are stored in ChromaDB for semantic retrieval

Retrieved context is later passed to a local LLM via Ollama

## Objective

This project demonstrates a production-ready local RAG pipeline, suitable for:

AI portfolio projects

Enterprise knowledge assistants

Secure / offline environments

Applied AI & MLOps learning
