"""
AI Knowledge Assistant - RAG Query with Ollama
Author: Ismael
Description: Query a local ChromaDB vector store and generate answers using a local LLM via Ollama.
"""

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# ---------- Step 1: Load embeddings ----------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------- Step 2: Load vector store ----------
vector_store = Chroma(
    collection_name="pdf_docs",
    embedding_function=embedding_model,
    persist_directory="./vectorstore/chroma"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ---------- Step 3: Load local LLM via Ollama ----------
llm = Ollama(
    model="mistral",
    temperature=0.2
)

# ---------- Step 4: Build RAG pipeline ----------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# ---------- Step 5: Interactive query ----------
while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    result = qa_chain(query)

    print("\nAnswer:")
    print(result["result"])

    print("\nSources:")
    for doc in result["source_documents"]:
        print("-", doc.metadata)
