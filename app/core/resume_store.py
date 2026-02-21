"""
app/core/resume_store.py — ChromaDB vector store for resume
"""
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.config import CHROMA_DB_PATH, OPENAI_API_KEY


def get_vectorstore() -> Chroma:
    """Returns the ChromaDB vector store (persistent)."""
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return Chroma(
        collection_name="resume",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )


def ingest_resume(resume_text: str) -> str:
    """
    Splits resume text into chunks and stores in ChromaDB.
    Call this once when user uploads/pastes their resume.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(resume_text)

    vectorstore = get_vectorstore()

    # Clear old resume data before re-ingesting
    vectorstore._collection.delete(where={"source": "resume"})

    vectorstore.add_texts(
        texts=chunks,
        metadatas=[{"source": "resume"} for _ in chunks],
    )

    return f"✅ Resume ingested: {len(chunks)} chunks stored."


def retrieve_resume_context(query: str, k: int = 5) -> str:
    """Retrieves the most relevant resume sections for a given query."""
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])
