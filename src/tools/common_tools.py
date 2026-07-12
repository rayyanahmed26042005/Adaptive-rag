"""
Common tools for document and description processing.
"""

from src.llms.gemini import llm


def enhance_description_with_llm(user_description: str) -> str:
    """
    Enhance user-provided document description using LLM.

    Rewrites the description to be suitable as a retriever tool instruction
    that clearly indicates the tool is only for answering questions about
    the uploaded content. If the LLM invocation fails, falls back to the
    original user description.

    Args:
        user_description: The original user-provided description.

    Returns:
        Enhanced description formatted as a tool instruction.
    """
    try:
        prompt = f"""
        Rewrite the following user-provided document description to be used as a retriever tool instruction.
        It should clearly state that the tool is only for answering questions about the uploaded content.

        Description: "{user_description}"

        Tool Instruction:"""

        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Warning: Failed to enhance description using LLM: {e}. Falling back to raw user description.")
        return user_description