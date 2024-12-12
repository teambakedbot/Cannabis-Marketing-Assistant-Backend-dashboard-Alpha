from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    Request,
    Path,
)
from typing import Optional
from ..services.chat_service import (
    process_chat_message,
    rename_chat,
    get_chat_messages,
    archive_chat,
    delete_chat,
    record_feedback,
    retry_message,
)
from ..services.user_service import get_user_chats
from ..services.auth_service import (
    get_firebase_user,
    get_current_user_optional,
)
import os
from ..crud.crud import (
    create_chat_session,
)
from ..models.schemas import (
    ChatRequest,
    ChatMessage,
    User,
    ChatSession,
    FeedbackCreate,
    MessageRetry,
)
from redis.asyncio import Redis
from ..config.config import logger
from ..utils.redis_config import get_redis


router = APIRouter(
    prefix="/api/v1",
    tags=["chat"],
    responses={404: {"description": "Not found"}},
)


async def handle_exception(e: Exception) -> HTTPException:
    """Helper function to handle exceptions and log errors."""
    logger.error(f"Error: {str(e)}")
    raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/chat", response_model=ChatMessage)
async def process_chat(
    request: Request,
    chat_request: ChatRequest,
    background_tasks: BackgroundTasks,
    redis: Redis = Depends(get_redis),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    try:
        session = request.session
        chat_id = chat_request.chat_id or session.get("chat_id") or os.urandom(16).hex()
        user_id = current_user.id if current_user else None
        client_ip = request.client.host
        voice_type = chat_request.voice_type
        session_id = session.get("session_id") or os.urandom(16).hex()
        message = chat_request.message
        user_agent = request.headers.get("User-Agent")

        response = await process_chat_message(
            user_id=user_id,
            chat_id=chat_id,
            session_id=session_id,
            client_ip=client_ip,
            message=message,
            user_agent=user_agent,
            voice_type=voice_type,
            background_tasks=background_tasks,
            redis_client=redis,
        )
        await background_tasks()
        return response
    except Exception as e:
        await handle_exception(e)


@router.get("/user/chats")
async def get_user_chats_endpoint(
    current_user: User = Depends(get_firebase_user),
    redis: Redis = Depends(get_redis),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    try:
        return await get_user_chats(current_user.id, redis, page, page_size)
    except Exception as e:
        await handle_exception(e)


@router.get("/chat/messages")
async def get_chat_messages_endpoint(
    chat_id: str = Query(..., description="The chat ID to fetch messages for"),
    current_user: User = Depends(get_firebase_user),
):
    try:
        return await get_chat_messages(chat_id)
    except Exception as e:
        await handle_exception(e)


@router.put("/chat/rename")
async def rename_chat_endpoint(
    chat_id: str = Query(...),
    new_name: str = Query(...),
    current_user: User = Depends(get_firebase_user),
):
    try:
        return await rename_chat(chat_id, new_name, current_user.id)
    except Exception as e:
        await handle_exception(e)


@router.put("/chat/{chat_id}/archive")
async def archive_chat_endpoint(
    chat_id: str = Path(...),
    current_user: User = Depends(get_firebase_user),
):
    try:
        return await archive_chat(chat_id, current_user.id)
    except Exception as e:
        await handle_exception(e)


@router.delete("/chat/{chat_id}")
async def delete_chat_endpoint(
    chat_id: str = Path(...),
    current_user: User = Depends(get_firebase_user),
):
    try:
        return await delete_chat(chat_id, current_user.id)
    except Exception as e:
        await handle_exception(e)


@router.post("/feedback")
async def record_feedback_endpoint(
    feedback: FeedbackCreate,
    current_user: User = Depends(get_firebase_user),
):
    """Record user feedback for a specific message."""
    try:
        result = await record_feedback(
            user_id=current_user.id,
            message_id=feedback.message_id,
            feedback_type=feedback.feedback_type,
        )
        return {"status": "success", "message": "Feedback recorded successfully"}
    except Exception as e:
        await handle_exception(e)


@router.post("/retry")
async def retry_message_endpoint(
    retry: MessageRetry,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_firebase_user),
    redis: Redis = Depends(get_redis),
):
    """Retry a specific message in the chat history."""
    try:
        result = await retry_message(
            user_id=current_user.id,
            message_id=retry.message_id,
            background_tasks=background_tasks,
            redis=redis,
        )
        return result
    except Exception as e:
        await handle_exception(e)


@router.post("/chat/start", response_model=ChatSession)
async def start_chat_session(
    current_user: User = Depends(get_firebase_user),
):
    try:
        return await create_chat_session(user_id=current_user.id)
    except Exception as e:
        await handle_exception(e)
