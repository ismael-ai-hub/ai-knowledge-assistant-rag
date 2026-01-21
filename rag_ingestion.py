"""
AI Knowledge Assistant - PDF Ingestion and Embeddings
Author: Ismael
Description: Ingest PDFs, split text into chunks, and create embeddings in ChromaDB
"""

# 1️⃣ Installer les packages nécessaires (si pas déjà)
# pip install langchain chromadb tiktoken PyPDF2

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from PyPDF2 import PdfReader

# ---------- Step 1: Load PDF ----------
pdf_path = "example.pdf"  # Remplace par ton PDF
reader = PdfReader(pdf_path)
raw_text = ""
for page in reader.pages:
    raw_text += page.extract_text()

# ---------- Step 2: Split text into chunks ----------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_text(raw_text)
print(f"Number of chunks: {len(chunks)}")

# ---------- Step 3: Create embeddings ----------
embeddings = OpenAIEmbeddings(openai_api_key="YOUR_OPENAI_API_KEY")
vector_store = Chroma.from_texts(chunks, embedding=embeddings, collection_name="pdf_docs")

print("PDF ingested and embeddings stored in ChromaDB successfully!")
