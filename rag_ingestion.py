"""
AI Knowledge Assistant - PDF Ingestion (Local & Free)
Author: Ismael
Description:
- Load PDF documents
- Split text into chunks
- Create embeddings using Hugging Face
- Store vectors in ChromaDB (local)
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pypdf import PdfReader
import os

# ---------- Configuration ----------
PDF_PATH = "data/example.pdf"   # place your PDF file here
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "pdf_docs"

# ---------- Step 1: Load PDF ----------
reader = PdfReader(PDF_PATH)
raw_text = ""

for page in reader.pages:
    if page.extract_text():
        raw_text += page.extract_text()

print("PDF loaded successfully.")

# ---------- Step 2: Split text ----------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_text(raw_text)
print(f"Number of text chunks: {len(chunks)}")

# ---------- Step 3: Create embeddings (Hugging Face) ----------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------- Step 4: Store in ChromaDB ----------
vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DB_DIR,
    collection_name=COLLECTION_NAME
)

vectorstore.persist()

print("PDF ingested and stored in ChromaDB successfully!")
