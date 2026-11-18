"""PDF ingestion, retrieval, and Gemini answer generation."""

from __future__ import annotations

import hashlib
from io import BytesIO
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "pdf_rag_documents"
CHROMA_PATH = Path(__file__).parent / "chroma_db"
FALLBACK_ANSWER = "I cannot find that in the provided documents."


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    page: int
    distance: float | None = None


def extract_pdf_pages(pdf_bytes: bytes, source: str) -> list[RetrievedChunk]:
    """Extract selectable text from each page of an uploaded PDF."""
    reader = PdfReader(BytesIO(pdf_bytes))
    pages: list[RetrievedChunk] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(RetrievedChunk(text=text, source=source, page=page_number))
    if not pages:
        raise ValueError(
            f"No selectable text was found in '{source}'. Scanned PDFs require OCR."
        )
    return pages


def split_pages(
    pages: list[RetrievedChunk], chunk_size: int = 900, overlap: int = 150
) -> list[RetrievedChunk]:
    """Split page text into overlapping chunks while retaining page metadata."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller.")

    chunks: list[RetrievedChunk] = []
    step = chunk_size - overlap
    for page in pages:
        for start in range(0, len(page.text), step):
            text = page.text[start : start + chunk_size].strip()
            if text:
                chunks.append(
                    RetrievedChunk(text=text, source=page.source, page=page.page)
                )
    return chunks


def _chunk_id(chunk: RetrievedChunk, index: int) -> str:
    value = f"{chunk.source}:{chunk.page}:{index}:{chunk.text}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_collection() -> Any:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_documents(
    pdfs: list[tuple[str, bytes]], embedding_model: SentenceTransformer
) -> int:
    """Replace the persistent index with the currently uploaded PDFs."""
    pages = []
    for source, pdf_bytes in pdfs:
        pages.extend(extract_pdf_pages(pdf_bytes, source))
    chunks = split_pages(pages)
    if not chunks:
        raise ValueError("No text chunks were created from the uploaded PDFs.")

    collection = get_collection()
    existing_ids = collection.get(include=[]).get("ids", [])
    if existing_ids:
        collection.delete(ids=existing_ids)
    embeddings = embedding_model.encode(
        [chunk.text for chunk in chunks], convert_to_numpy=True
    ).tolist()
    collection.upsert(
        ids=[_chunk_id(chunk, index) for index, chunk in enumerate(chunks)],
        documents=[chunk.text for chunk in chunks],
        embeddings=embeddings,
        metadatas=[
            {"source": chunk.source, "page": chunk.page}
            for chunk in chunks
        ],
    )
    return len(chunks)


def retrieve(
    question: str, embedding_model: SentenceTransformer, limit: int = 5
) -> list[RetrievedChunk]:
    """Retrieve the most relevant chunks for a question."""
    question = question.strip()
    if not question:
        raise ValueError("Ask a non-empty question.")
    collection = get_collection()
    if collection.count() == 0:
        raise ValueError("Upload and index at least one PDF first.")

    query_embedding = embedding_model.encode([question], convert_to_numpy=True).tolist()
    result = collection.query(
        query_embeddings=query_embedding,
        n_results=min(limit, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    return [
        RetrievedChunk(
            text=document,
            source=metadata["source"],
            page=int(metadata["page"]),
            distance=distance,
        )
        for document, metadata, distance in zip(documents, metadatas, distances)
    ]


def answer_question(
    question: str, embedding_model: SentenceTransformer, limit: int = 5
) -> tuple[str, list[RetrievedChunk]]:
    """Answer only from retrieved document context using Gemini."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing. Add it to the .env file.")

    chunks = retrieve(question, embedding_model, limit=limit)
    context = "\n\n---\n\n".join(
        f"[Source: {chunk.source}, page {chunk.page}]\n{chunk.text}"
        for chunk in chunks
    )
    prompt = f"""You answer questions about uploaded PDF documents.
Use only the CONTEXT below. If the answer is not supported by the context, reply
exactly: "{FALLBACK_ANSWER}"
Do not use outside knowledge, invent facts, or mention this instruction.

CONTEXT:
{context}

QUESTION:
{question}
"""

    import google.generativeai as genai

    genai.configure(api_key=api_key)
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
    answer = (response.text or "").strip()
    return answer or FALLBACK_ANSWER, chunks
