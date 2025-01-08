from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from fastapi import HTTPException
from google.cloud import firestore
from redis.asyncio import Redis
from langchain_core.messages import HumanMessage, AIMessage
from ..config.config import logger, settings
from ..utils.firebase_utils import db
from ..models.schemas import ChatMessage
from .firestore_chat_history import FirestoreChatMessageHistory


async def generate_default_chat_name(message: str) -> str:
    """Generate a default name for a new chat based on the first message."""
    title_prompt = f"Generate a short, catchy title (max 5 words) for a chat that starts with this message: '{message}'"
    messages = [HumanMessage(content=title_prompt)]
    response = await settings.llm.ainvoke(messages)
    return response.content.strip().strip("\"'")


async def get_cached_chat_name(chat_id: str, redis_client: Redis) -> str:
    """Get the chat name from cache or Firestore."""
    cached_name = await redis_client.get(f"chat_name:{chat_id}")
    if cached_name:
        return cached_name.decode("utf-8")

    chat_ref = db.collection("chats").document(chat_id)
    chat_doc = await chat_ref.get()
    name = (
        chat_doc.to_dict().get("name", "Unnamed Chat")
        if chat_doc.exists
        else "Unnamed Chat"
    )

    await redis_client.set(f"chat_name:{chat_id}", name, ex=3600)
    return name


async def rename_chat(chat_id: str, new_name: str, user_id: str) -> Dict[str, str]:
    """Rename a chat if the user has permission."""
    try:
        if not new_name:
            raise HTTPException(status_code=400, detail="New name must be provided")

        chat_ref = db.collection("chats").document(chat_id)
        chat_doc = await chat_ref.get()

        if not chat_doc.exists:
            raise HTTPException(status_code=404, detail="Chat not found")

        chat_data = chat_doc.to_dict()
        if user_id not in chat_data.get("user_ids", []):
            raise HTTPException(
                status_code=403, detail="User not authorized to rename this chat"
            )

        await chat_ref.update(
            {"name": new_name, "updated_at": firestore.SERVER_TIMESTAMP}
        )

        logger.debug(f"Chat {chat_id} renamed to {new_name} by user {user_id}")
        return {"message": "Chat renamed successfully"}

    except HTTPException as http_ex:
        logger.error(f"HTTP error occurred: {http_ex.detail}")
        raise
    except Exception as e:
        logger.error(f"Error occurred while renaming chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def archive_chat(chat_id: str, user_id: str) -> Dict[str, str]:
    """Archive a chat if the user has permission."""
    try:
        chat_ref = db.collection("chats").document(chat_id)
        chat_doc = await chat_ref.get()

        if not chat_doc.exists:
            raise HTTPException(status_code=404, detail="Chat not found")

        chat_data = chat_doc.to_dict()
        if user_id not in chat_data.get("user_ids", []):
            raise HTTPException(
                status_code=403, detail="User not authorized to archive this chat"
            )

        await chat_ref.update(
            {"archived": True, "updated_at": firestore.SERVER_TIMESTAMP}
        )

        logger.debug(f"Chat {chat_id} archived by user {user_id}")
        return {"message": "Chat archived successfully"}

    except Exception as e:
        logger.error(f"Error occurred while archiving chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def delete_chat(chat_id: str, user_id: str) -> Dict[str, str]:
    """Delete a chat and its messages if the user has permission."""
    try:
        chat_ref = db.collection("chats").document(chat_id)
        chat_doc = await chat_ref.get()

        if not chat_doc.exists:
            raise HTTPException(status_code=404, detail="Chat not found")

        chat_data = chat_doc.to_dict()
        if user_id not in chat_data.get("user_ids", []):
            raise HTTPException(
                status_code=403, detail="User not authorized to delete this chat"
            )

        messages_ref = chat_ref.collection("messages")
        batch = db.batch()
        messages = await messages_ref.get()

        for msg in messages:
            batch.delete(msg.reference)
        batch.delete(chat_ref)
        await batch.commit()

        logger.debug(f"Chat {chat_id} and its messages deleted by user {user_id}")
        return {"message": "Chat deleted successfully"}

    except Exception as e:
        logger.error(f"Error occurred while deleting chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_chat_messages(chat_id: str) -> Dict[str, Any]:
    """Get all messages for a chat."""
    logger.debug(f"Fetching chat messages for chat_id: {chat_id}")
    try:
        chat_history = FirestoreChatMessageHistory(chat_id)
        messages = await chat_history.get_messages()
        return {"chat_id": chat_id, "messages": messages}
    except Exception as e:
        logger.error(f"Error occurred while fetching chat messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def record_feedback(
    user_id: str, message_id: str, feedback_type: str
) -> Dict[str, str]:
    """Record user feedback for a message."""
    feedback_ref = db.collection("feedback").document()
    feedback_data = {
        "user_id": user_id,
        "message_id": message_id,
        "feedback_type": feedback_type,
        "timestamp": firestore.SERVER_TIMESTAMP,
    }
    await feedback_ref.set(feedback_data)
    return {"status": "success", "feedback_id": feedback_ref.id}


async def update_chat_document(
    user_id: Optional[str],
    chat_id: str,
    session_id: str,
    message: str,
    redis_client: Redis,
) -> None:
    """Update or create a chat document with user and session information."""
    chat_ref = db.collection("chats").document(chat_id)
    chat_doc = await chat_ref.get()

    if not chat_doc.exists:
        default_name = await generate_default_chat_name(message)
        chat_data = {
            "chat_id": chat_id,
            "user_ids": [user_id] if user_id else [],
            "session_ids": [session_id] if not user_id else [],
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
            "name": default_name,
        }
        await chat_ref.set(chat_data)
        cache_key = f"user_chats:{user_id}:page:{1}:size:{20}"
        await redis_client.delete(cache_key)
        logger.debug(f"New chat document created with chat_id: {chat_id}")
    else:
        updates = {}
        if user_id and user_id not in chat_doc.to_dict().get("user_ids", []):
            updates["user_ids"] = firestore.ArrayUnion([user_id])
        if not user_id and session_id not in chat_doc.to_dict().get("session_ids", []):
            updates["session_ids"] = firestore.ArrayUnion([session_id])
        if updates:
            updates["updated_at"] = firestore.SERVER_TIMESTAMP
            await chat_ref.update(updates)
            logger.debug(f"Chat document updated with chat_id: {chat_id}")


async def manage_session(
    session_id: str,
    user_agent: str,
    client_ip: str,
    user_id: Optional[str],
) -> None:
    """Create or update a session document."""
    session_ref = db.collection("sessions").document(session_id)
    session_doc = await session_ref.get()

    if not session_doc.exists:
        session_data = {
            "session_id": session_id,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "user_agent": user_agent,
            "ip_address": client_ip,
            "user_id": user_id if user_id else None,
        }
        await session_ref.set(session_data, merge=True)
    elif user_id and user_id != session_doc.to_dict().get("user_id"):
        await session_ref.update({"user_id": user_id})


def is_valid_email_domain(email: str) -> bool:
    """Check if email is from an allowed domain."""
    return not ("test" in email.lower() or email.lower().endswith("@bakedbot.ai"))


async def get_monthly_chat_stats(
    user_id: Optional[str] = None,
    user_email: Optional[str] = None,
) -> Dict[str, Any]:
    """Get monthly chat statistics and summaries."""
    try:
        today = datetime.now()
        first_day = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        chats_ref = db.collection("chats")

        if user_id:
            chats = await chats_ref.where("user_ids", "array_contains", user_id).get()
        else:
            chats = await chats_ref.get()

        monthly_stats = {
            "total_messages": 0,
            "chat_summaries": [],
            "email": user_email if user_email else "anonymous",
            "skipped": False,
        }

        if user_email and not is_valid_email_domain(user_email):
            logger.info(f"Skipping stats for test/internal email: {user_email}")
            return {
                "total_messages": 0,
                "chat_summaries": [],
                "email": user_email,
                "skipped": True,
                "reason": "Test or internal email address",
            }

        for chat in chats:
            messages_ref = chat.reference.collection("messages")
            messages = (
                await messages_ref.where("timestamp", ">=", first_day)
                .order_by("timestamp")
                .get()
            )

            if not messages:
                continue

            message_count = len(messages)
            monthly_stats["total_messages"] += message_count

            chat_content = " ".join(
                f"{msg.get('role')}: {msg.get('content')}"
                for msg in [msg.to_dict() for msg in messages]
            )

            summary_prompt = [
                AIMessage(
                    content="You are a helpful assistant that creates very brief (1-2 sentences) summaries of conversations."
                ),
                HumanMessage(
                    content=f"Please summarize this conversation briefly: {chat_content}"
                ),
            ]
            summary_response = await settings.llm.ainvoke(summary_prompt)

            chat_data = chat.to_dict()
            monthly_stats["chat_summaries"].append(
                {
                    "chat_id": chat.id,
                    "name": chat_data.get("name", "Unnamed Chat"),
                    "message_count": message_count,
                    "summary": summary_response.content,
                    "last_message": messages[-1].get("timestamp"),
                }
            )

        monthly_stats["chat_summaries"].sort(
            key=lambda x: x["last_message"], reverse=True
        )

        return monthly_stats

    except Exception as e:
        logger.error(f"Error getting monthly chat stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
