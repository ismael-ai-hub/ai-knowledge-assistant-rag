import sys
sys.stdout.reconfigure(encoding="utf-8")

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "pdf_docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

app = FastAPI(title="Local RAG API", version="1.0")

# --- Load components once (startup) ---
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

llm = OllamaLLM(model="mistral", temperature=0.2)

prompt = ChatPromptTemplate.from_template("""
Tu es un assistant pédagogique.

Règles obligatoires :
- Réponds UNIQUEMENT à la question de l'utilisateur (pas aux questions présentes dans le document).
- N'invente rien. Utilise uniquement le CONTEXTE fourni.
- Si la réponse n'est pas dans le contexte, réponds : "Je ne sais pas."
- Si l'utilisateur demande "oui ou non", réponds uniquement par "Oui" ou "Non" (un seul mot).

CONTEXTE :
{context}

QUESTION UTILISATEUR :
{question}

RÉPONSE :
""")


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    question = req.question.strip()
    yes_no_mode = "oui ou non" in question.lower() or "yes or no" in question.lower()

    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    answer = llm.invoke(
        prompt.format(context=context, question=question)
    )

    # Yes/No post-processing (same logic as CLI)
    if yes_no_mode:
        ans = answer.strip().lower()
        if "oui" in ans and "non" in ans:
            answer = "Non"
        elif "oui" in ans:
            answer = "Oui"
        elif "non" in ans:
            answer = "Non"
        else:
            answer = "Je ne sais pas."

    sources = []
    seen = set()
    for d in docs:
        src = d.metadata.get("source", "unknown")
        ch = d.metadata.get("chunk", "?")
        key = (src, ch)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source": src, "chunk": ch})
        if len(sources) == 3:
            break

    return {"answer": answer, "sources": sources}
