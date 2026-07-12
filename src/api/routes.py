"""
API routes for RAG operations.
"""

import hashlib
import uuid

from fastapi import APIRouter, UploadFile, File, Header, HTTPException, status
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

from src.db.mongo_client import db
from src.memory.chat_history_mongo import ChatHistory
from src.models.query_request import QueryRequest
from src.rag.document_upload import documents
from src.rag.graph_builder import builder

router = APIRouter()


class UserAuth(BaseModel):
    """Schema for user authentication request."""

    username: str
    password: str


# MongoDB collections for authentication
users_collection = db["users"]
tokens_collection = db["api_tokens"]


@router.post("/rag/query")
async def rag_query(req: QueryRequest):
    """
    Process a RAG query and return the result.

    Args:
        req: The query request containing query text and session_id.

    Returns:
        The generated response from the RAG pipeline.
    """
    try:
        #chat_history=ChatInMemoryHistory.get_session_history(req.token)
        chat_history = ChatHistory.get_session_history(req.session_id)
        await chat_history.add_message(HumanMessage(content=req.query))

        # Fetch full history
        messages = await chat_history.get_messages()
        result = builder.invoke({
            "messages": messages
        })
        output_text = result["messages"][-1].content

        # Save assistant message
        await chat_history.add_message(AIMessage(content=output_text))

        return {"result": result["messages"][-1]}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing RAG query: {str(e)}"
        )


@router.post("/rag/documents/upload")
async def upload_file(
    file: UploadFile = File(...),
    description: str = Header(..., alias="X-Description")
):
    """
    Upload a document for RAG processing.

    Args:
        file: The file to upload (PDF or TXT).
        description: Document description provided via header.

    Returns:
        Upload status.
    """
    try:
        status_upload = documents(description, file)
        return {"status": status_upload}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document upload: {str(e)}"
        )


@router.post("/api/init")
async def api_init():
    """
    Initialize API token session.

    Returns:
        A dictionary containing the generated API token.
    """
    token = str(uuid.uuid4())
    await tokens_collection.insert_one({"api_token": token})
    return {"api_token": token}


@router.post("/api/create_user")
async def create_user_endpoint(
    req: UserAuth,
    x_api_token: str = Header(..., alias="X-API-TOKEN")
):
    """
    Register a new user account.

    Args:
        req: Username and password.
        x_api_token: API token header validation.

    Returns:
        Status indicating success.
    """
    # Verify API token
    token_exists = await tokens_collection.find_one({"api_token": x_api_token})
    if not token_exists:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token"
        )

    # Check if user already exists
    existing_user = await users_collection.find_one({"username": req.username})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User already exists"
        )

    # Hash password using SHA-256
    hashed_password = hashlib.sha256(req.password.encode()).hexdigest()

    # Save user
    await users_collection.insert_one({
        "username": req.username,
        "password": hashed_password
    })
    return {"status": "success"}


@router.post("/api/login")
async def login_user_endpoint(
    req: UserAuth,
    x_api_token: str = Header(..., alias="X-API-TOKEN")
):
    """
    Authenticate user login.

    Args:
        req: Username and password.
        x_api_token: API token header validation.

    Returns:
        JWT token on success.
    """
    # Verify API token
    token_exists = await tokens_collection.find_one({"api_token": x_api_token})
    if not token_exists:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token"
        )

    # Find user
    user = await users_collection.find_one({"username": req.username})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    # Verify password
    hashed_password = hashlib.sha256(req.password.encode()).hexdigest()
    if user["password"] != hashed_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    # Generate session/jwt token
    jwt_token = f"session_{uuid.uuid4()}"
    return {"jwt": jwt_token}


