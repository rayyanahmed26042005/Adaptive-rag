"""
API client for communicating with backend services.
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)

# Backend service URLs
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
RUST_BASE_URL = f"{BACKEND_URL}/api"
PYTHON_BASE_URL = BACKEND_URL


def create_user(username: str, password: str, api_token: str) -> bool:
    """
    Create a new user account.

    Args:
        username: Username for the new account.
        password: Password for the new account.
        api_token: API token for authentication.

    Returns:
        True if user creation succeeds, False otherwise.
    """
    headers = {
        "X-API-TOKEN": api_token,
        "Content-Type": "application/json"
    }
    logger.info("API Token received: %s", api_token)

    try:
        response = requests.post(
            f"{RUST_BASE_URL}/create_user",
            json={"username": username, "password": password},
            headers=headers,
        )

        logger.info("Calling /create_user, status code: %s", response.status_code)

        if response.status_code == 200:
            try:
                logger.debug("Create user response: %s", response.json())
            except ValueError:
                logger.warning("Create user returned non-JSON response")
            return True
        else:
            logger.error(
                "Create user failed: %s - %s",
                response.status_code,
                response.text
            )
            return False

    except requests.RequestException as e:
        logger.exception("Request to /create_user failed: %s", e)
        return False


def login_user(username: str, password: str, api_token: str) -> dict:
    """
    Authenticate user login.

    Args:
        username: Username to log in.
        password: Password for the user.
        api_token: API token for authentication.

    Returns:
        Response dictionary with JWT token if successful, None otherwise.
    """
    headers = {
        "X-API-TOKEN": api_token,
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(
            f"{RUST_BASE_URL}/login",
            json={"username": username, "password": password},
            headers=headers,
        )
        logger.info("Calling /login, status code: %s", response.status_code)

        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logger.exception("Login request failed: %s", e)

    return None


def get_api_token() -> str:
    """
    Get an API token for authentication.

    Returns:
        API token string if successful, None otherwise.
    """
    try:
        response = requests.post(f"{RUST_BASE_URL}/init")
        logger.info("Calling /init, status code: %s", response.status_code)

        if response.status_code == 200:
            return response.json()["api_token"]
    except Exception as e:
        logger.exception("Failed to initialize API token: %s", e)

    return None


def query_backend(query: str, session_id: str) -> str:
    """
    Send a query to the RAG backend.

    Args:
        query: The user's query text.
        session_id: Session identifier for tracking conversation.

    Returns:
        Response text from the backend or error message.
    """
    url = f"{PYTHON_BASE_URL}/rag/query"
    print(f"[query_backend] Calling: {url}")

    try:
        response = requests.post(
            url,
            json={"query": query, "session_id": session_id},
            allow_redirects=False
        )

        if response.status_code == 200:
            return response.json()["result"]["content"]
        else:
            try:
                error_detail = response.json().get("detail", response.text)
            except Exception:
                error_detail = response.text
            return f"Error ({response.status_code}): {error_detail}"
    except Exception as e:
        return f"Error: Failed to connect to backend: {str(e)}"


def document_upload_rag(file, description: str) -> tuple[bool, str]:
    """
    Upload a document to the RAG system.

    Args:
        file: File object to upload.
        description: Description of the document.

    Returns:
        Tuple of (success_status, status_message).
    """
    headers = {
        "X-Description": description
    }
    url = f"{PYTHON_BASE_URL}/rag/documents/upload"

    if file:
        try:
            files = {"file": (file.name, file, file.type)}
            response = requests.post(url, files=files, headers=headers)
            print(response)

            if response.status_code == 200:
                return True, "Successfully uploaded and indexed document."
            else:
                try:
                    error_detail = response.json().get("detail", response.text)
                except Exception:
                    error_detail = response.text
                return False, f"Server error ({response.status_code}): {error_detail}"
        except Exception as e:
            return False, f"Connection error: {str(e)}"

    return False, "No file was provided."
