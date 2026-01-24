"""
AI Knowledge Assistant - PDF Ingestion & Embeddings
Author: Ismael
Description:
- Load PDF documents
- Split text into semantic chunks
- Generate embeddings using Hugging Face
- Store vectors locally in ChromaDB
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from PyPDF2 import PdfReader
import os

# ---------- Configuration ----------
DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------- Step 1: Load PDFs ----------
documents = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(DATA_DIR, file)
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        documents.append(text)

print(f"Loaded {len(documents)} PDF(s)")

# ---------- Step 2: Split text ----------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = []
for doc in documents:
    chunks.extend(text_splitter.split_text(doc))

print(f"Generated {len(chunks)} text chunks")

# ---------- Step 3: Embeddings (Hugging Face) ----------
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

# ---------- Step 4: Store in ChromaDB ----------
vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    persist_directory=VECTORSTORE_DIR
)

vectorstore.persist()

print("Documents successfully ingested and stored in ChromaDB")
