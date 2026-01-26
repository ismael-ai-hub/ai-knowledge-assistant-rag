"""
AI Knowledge Assistant - RAG Query with Ollama
Author: Ismael
Description: Query a local ChromaDB vector store and generate answers using a local LLM via Ollama.
"""


from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

import sys
sys.stdout.reconfigure(encoding="utf-8")

CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "pdf_docs"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_DIR
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

llm = OllamaLLM(
    model="mistral",
    temperature=0.2
)

prompt = ChatPromptTemplate.from_template("""
Tu es un assistant pédagogique.

Règles obligatoires :
- Réponds UNIQUEMENT à la question de l'utilisateur (pas aux questions présentes dans le document).
- N'invente rien. Utilise uniquement le CONTEXTE fourni.
- Si la réponse n'est pas dans le contexte, réponds : "Je ne sais pas."
- Si l'utilisateur demande "oui ou non", réponds uniquement par "Oui" ou "Non" (un seul mot).

HISTORIQUE (pour comprendre la conversation) :
{history}

CONTEXTE :
{context}

QUESTION UTILISATEUR :
{question}

RÉPONSE :
""")

history = []  # list of tuples (user_question, assistant_answer)

while True:
    query = input("\nAsk a question (or type 'exit'): ").strip()
    if query.lower() == "exit":
        break

    # Détection oui / non
    yes_no_mode = "oui ou non" in query.lower() or "yes or no" in query.lower()

    # Récupération des documents
    docs = retriever.invoke(query)

    print("\nTop sources used:")
    seen = set()
    shown = 0
    for d in docs:
        src = d.metadata.get("source", "unknown")
        ch = d.metadata.get("chunk", "?")
        key = (src, ch)
        if key in seen:
            continue
        seen.add(key)

        snippet = (d.page_content[:180] + "...").replace("\n", " ")
        print(f"- {src} (chunk {ch}): {snippet}")
        shown += 1
        if shown == 3:
            break

    # Construction du contexte
    context = "\n\n".join(doc.page_content for doc in docs)

    # Historique (mémoire)
    history_text = "\n".join(
        [f"User: {u}\nAssistant: {a}" for u, a in history][-6:]
    )

    # Appel du LLM
    answer = llm.invoke(
        prompt.format(
            context=context,
            question=query,
            history=history_text
        )
    )

    # Post-traitement oui / non
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

    # Sauvegarde dans l'historique
    history.append((query, answer))

    # Affichage final
    print("\nAnswer:\n", answer)
