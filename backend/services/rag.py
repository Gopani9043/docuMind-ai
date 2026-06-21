import os
import logging
import numpy as np
import faiss
import pickle
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Lazy model loader — only loads when first needed ──────────────────────────
_embedder = None

def get_embedder():
    """
    Load SentenceTransformer only when actually needed.
    Never loads on import — fixes backend startup hang.
    """
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading SentenceTransformer model...")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("SentenceTransformer model loaded.")
    return _embedder


llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
    temperature=0,
)

# ── Storage path ──────────────────────────────────
VECTOR_STORE_DIR = Path(__file__).parent.parent / "vector_stores"
VECTOR_STORE_DIR.mkdir(exist_ok=True)

GENERAL_QA_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant for document processing.
Answer the following question clearly and concisely.

QUESTION:
{question}

Answer:
""")

DOCUMENT_QA_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful document assistant.
Answer the question based ONLY on the document context below.
Answer in one clear sentence. No brackets, no source references, no explanations.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

Answer:
""")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Split text into overlapping chunks for better context."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def index_document(doc_id: str, raw_text: str) -> None:
    """
    Convert document text to vectors and save FAISS index.
    Called after OCR extraction — runs in Celery worker, not backend.
    """
    try:
        chunks = chunk_text(raw_text)
        if not chunks:
            logger.warning(f"No chunks for doc {doc_id}")
            return

        logger.info(f"Indexing {len(chunks)} chunks for doc {doc_id}")

        # ── Use lazy loader — model only loads here in Celery ──
        embedder = get_embedder()
        embeddings = embedder.encode(chunks, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")

        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        index_path  = VECTOR_STORE_DIR / f"{doc_id}.index"
        chunks_path = VECTOR_STORE_DIR / f"{doc_id}.chunks"

        faiss.write_index(index, str(index_path))
        with open(chunks_path, "wb") as f:
            pickle.dump(chunks, f)

        logger.info(f"Indexed doc {doc_id} successfully")

    except Exception as e:
        logger.error(f"Indexing failed for {doc_id}: {e}")
        raise


def search_document(doc_id: str, query: str, top_k: int = 3) -> list:
    """Search for relevant chunks in a document."""
    index_path  = VECTOR_STORE_DIR / f"{doc_id}.index"
    chunks_path = VECTOR_STORE_DIR / f"{doc_id}.chunks"

    if not index_path.exists():
        raise FileNotFoundError(f"No index found for document {doc_id}")

    index = faiss.read_index(str(index_path))
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    # ── Use lazy loader ──
    embedder = get_embedder()
    query_vec = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(query_vec)

    _, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


def answer_with_document(doc_id: str, question: str) -> dict:
    """Answer a question using RAG on a specific document."""
    try:
        relevant_chunks = search_document(doc_id, question)
        context = "\n\n---\n\n".join(relevant_chunks)

        chain = DOCUMENT_QA_PROMPT | llm
        response = chain.invoke({
            "context": context,
            "question": question
        })

        return {
            "answer": response.content,
            "mode": "document",
            "doc_id": doc_id,
            "sources": relevant_chunks[:2]
        }

    except FileNotFoundError:
        return {
            "answer": "This document has not been indexed yet. Please re-upload it.",
            "mode": "error",
            "doc_id": doc_id,
            "sources": []
        }
    except Exception as e:
        logger.error(f"RAG failed: {e}")
        raise


def answer_general(question: str) -> dict:
    """Answer a general question without document context."""
    chain = GENERAL_QA_PROMPT | llm
    response = chain.invoke({"question": question})
    return {
        "answer": response.content,
        "mode": "general",
        "doc_id": None,
        "sources": []
    }