from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from ..models.schemas import ChatRequest, GemmaChatResponse
from ..services.gemma_service import GemmaChatService
from ..config.config import logger

router = APIRouter()

# Initialize the service at module level
try:
    gemma_service = GemmaChatService()
except Exception as e:
    logger.error(f"Failed to initialize GemmaChatService: {e}")
    gemma_service = None


@router.post("/gemma/chat", response_model=GemmaChatResponse)
async def chat_with_gemma(
    request: ChatRequest, max_length: Optional[int] = 256
) -> GemmaChatResponse:
    """
    Chat with the GemmaLM Cannabis model

    Args:
        request: ChatRequest containing the message
        max_length: Optional maximum length for the response

    Returns:
        GemmaChatResponse: Contains the model's response
    """
    if gemma_service is None:
        raise HTTPException(
            status_code=503, detail="Cannabis language model service is not available"
        )

    try:
        response = await gemma_service.generate_response(
            prompt=request.message, max_length=max_length
        )

        return GemmaChatResponse(response=response, model="GemmaLM-for-Cannabis")
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing chat request: {str(e)}"
        )
