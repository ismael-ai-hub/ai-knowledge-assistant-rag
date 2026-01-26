"""
AI Knowledge Assistant - PDF Ingestion (Local & Free)
Author: Ismael
Description:
- Load PDF documents
- Split text into chunks
- Create embeddings using Hugging Face
- Store vectors in ChromaDB (local)
"""

import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DOCS_DIR = "data/docs"
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "pdf_docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_all_pdfs_text(folder: str) -> list[tuple[str, str]]:
    items = []
    for name in os.listdir(folder):
        if name.lower().endswith(".pdf"):
            path = os.path.join(folder, name)
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                t = page.extract_text() or ""
                text += t + "\n"
            items.append((path, text))
    return items

def main():
    pdfs = load_all_pdfs_text(DOCS_DIR)
    if not pdfs:
        raise SystemExit(f"No PDFs found in {DOCS_DIR}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    texts = []
    metas = []

    for source_path, raw_text in pdfs:
        chunks = splitter.split_text(raw_text)
        for i, c in enumerate(chunks):
            texts.append(c)
            metas.append({"source": source_path, "chunk": i})

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    import shutil

    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)


    Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metas,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_DIR,
    )
    print(f"Ingested {len(pdfs)} PDF(s) -> {len(texts)} chunks into {CHROMA_DB_DIR}/{COLLECTION_NAME}")

if __name__ == "__main__":
    main()
