"""
Retriever setup and vector store configuration.
"""

import os

from langchain_core.documents import Document
from langchain_core.tools import Tool, create_retriever_tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_qdrant import QdrantVectorStore
from langchain_community.vectorstores import FAISS

from src.core.config import settings

api_key = os.getenv("GOOGLE_API_KEY") or "placeholder_key"
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)

# Global variable to store the FAISS vectorstore instance
# This ensures get_retriever() can access documents stored by retriever_chain()
_faiss_vectorstore = None


def retriever_chain(chunks: list[Document]):
    """
    Initialize and store documents in FAISS vector database.

    Args:
        chunks: List of document chunks to store.

    Returns:
        Boolean indicating success of the operation.
    """
    global _faiss_vectorstore

    try:
        # Commenting out Qdrant code for temporary FAISS usage
        # vectorstore = QdrantVectorStore.from_documents(
        #     documents=chunks,
        #     embedding=embeddings,
        #     url=settings.QDRANT_URL,
        #     api_key=settings.QDRANT_API_KEY,
        #     collection_name=settings.CODE_COLLECTION,
        # )
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        # Store the vectorstore globally so get_retriever() can access it
        _faiss_vectorstore = vectorstore

        print("FAISS vector store initialized with documents")
        print(f"Vectorstore contains {len(chunks)} document chunks")
        return True
    except Exception as e:
        print(f"Error storing documents in FAISS: {e}")
        raise e


def search_uploaded_documents(query: str) -> str:
    """
    Search the uploaded documents for relevant information.

    Args:
        query: The search query.

    Returns:
        Concatenated document contents or a message if no documents are uploaded.
    """
    global _faiss_vectorstore
    if _faiss_vectorstore is None:
        return "No documents have been uploaded yet. Please upload a document first."
    retriever = _faiss_vectorstore.as_retriever()
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


def get_retriever():
    """
    Get a retriever tool connected to the FAISS vector store.

    Returns:
        A LangChain Tool configured for the vector store.
    """
    if os.path.exists("description.txt"):
        with open("description.txt", "r", encoding="utf-8") as f:
            description = f.read().strip()
    else:
        description = "uploaded documents"

    return Tool(
        name="retriever_customer_uploaded_documents",
        description=(
            f"Use this tool **only** to answer questions about: {description}\n"
            "Don't use this tool to answer anything else."
        ),
        func=search_uploaded_documents
    )
