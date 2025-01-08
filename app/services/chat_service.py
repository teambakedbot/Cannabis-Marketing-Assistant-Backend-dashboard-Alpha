from datetime import datetime
from typing import Optional, AsyncGenerator, Dict, Any, TypedDict, List
from fastapi import HTTPException

# Core components
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import StreamingStdOutCallbackHandler

# LangGraph components
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Main langchain package
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferMemory

from ..config.config import logger, settings
from ..models.schemas import ChatMessage
from ..utils.firebase_utils import db
from ..tools.tools import configurable_agent, llm
from ..utils.sessions import clean_up_old_sessions
from ..utils.redis_config import FirestoreEncoder

from firebase_admin import firestore
from redis.asyncio import Redis
import json
import os
import asyncio
from functools import lru_cache
from fastapi import BackgroundTasks


class MessagesState(TypedDict):
    """State definition for the chat graph."""

    messages: List[Dict[str, str]]
    metadata: Dict[str, Any]


class FirestoreChatMessageHistory:
    def __init__(self, chat_id: str):
        self.chat_id = chat_id
        self.collection = (
            db.collection("chats").document(chat_id).collection("messages")
        )

    async def add_message(self, message: ChatMessage):
        await self.collection.add(message.dict())

    async def get_messages(self, limit: Optional[int] = None):
        query = self.collection.order_by("timestamp")
        if limit:
            query = query.limit(limit)
        messages = await query.get()
        return [self._dict_to_message(msg.to_dict()) for msg in messages]

    def _dict_to_message(self, msg_dict: Dict) -> ChatMessage:
        return ChatMessage(
            content=msg_dict["content"],
            role=msg_dict["role"],
            message_id=msg_dict.get("message_id"),
            timestamp=msg_dict.get("timestamp"),
            session_id=msg_dict.get("session_id"),
            chat_id=msg_dict.get("chat_id"),
            data=msg_dict.get("data"),
            user_id=msg_dict.get("user_id"),
        )

    async def clear(self):
        batch = db.batch()
        docs = await self.collection.get()
        for doc in docs:
            batch.delete(doc.reference)
        await batch.commit()


async def process_chat_message(
    user_id: Optional[str],
    chat_id: str,
    session_id: str,
    client_ip: str,
    message: str,
    user_agent: str,
    voice_type: str,
    background_tasks: BackgroundTasks,
    redis_client: Redis,
    language: str = "English",
) -> AsyncGenerator[ChatMessage, None]:
    """Process a chat message and stream the AI response."""
    try:
        chat_history = FirestoreChatMessageHistory(chat_id)

        # Create user message
        user_message = ChatMessage(
            message_id=os.urandom(16).hex(),
            user_id=user_id,
            session_id=session_id,
            role="human",
            content=message,
            chat_id=chat_id,
            timestamp=datetime.utcnow(),
        )

        # Add user message to history
        await chat_history.add_message(user_message)

        # Update chat metadata
        await update_chat_document(user_id, chat_id, session_id, message, redis_client)
        await manage_session(session_id, user_agent, client_ip, user_id)

        # Configure voice and language
        voice_prompts = {
            "normal": "You are an AI-powered chatbot specialized in assisting cannabis marketers. Your name is Smokey.",
            "pops": "You are a fatherly and upbeat AI assistant, ready to help with cannabis marketing. But you sound like Pops from the movie Friday, use his style of talk.",
            "smokey": "You are a laid-back and cool AI assistant, providing cannabis marketing insights. But sounds like Smokey from the movie Friday, use his style of talk.",
        }
        voice_prompt = voice_prompts.get(voice_type, voice_prompts["normal"])

        # Create streaming callback handler
        class StreamingHandler(StreamingStdOutCallbackHandler):
            def __init__(self):
                super().__init__()
                self.tokens = []

            async def on_llm_new_token(self, token: str, **kwargs) -> None:
                self.tokens.append(token)
                # Yield partial message
                yield ChatMessage(
                    chat_id=chat_id,
                    content="".join(self.tokens),
                    role="ai",
                    timestamp=datetime.utcnow(),
                )

        handler = StreamingHandler()

        # Configure agent
        config = RunnableConfig(
            callbacks=[handler],
            configurable={
                "thread_id": chat_id,
                "user_id": user_id,
                "session_id": session_id,
                "language": language,
            },
        )

        # Process message with agent
        result = await configurable_agent.ainvoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
        )

        # Process response
        ai_response = result["messages"][-1]["content"]
        response_text = ""
        data = None

        # Handle tuple response from product recommendation
        if isinstance(ai_response, tuple):
            response_text, product_data = ai_response
            try:
                data = {"products": json.loads(product_data).get("products", [])}
            except json.JSONDecodeError:
                logger.error(f"Error decoding product data: {product_data}")
                data = None
        else:
            response_text = str(ai_response)

        # Create assistant message
        assistant_message = ChatMessage(
            message_id=os.urandom(16).hex(),
            user_id=None,
            session_id=session_id,
            role="ai",
            content=response_text,
            chat_id=chat_id,
            data=data,
            timestamp=datetime.utcnow(),
        )

        # Add assistant message to history
        await chat_history.add_message(assistant_message)

        # Schedule cleanup
        background_tasks.add_task(clean_up_old_sessions)

        yield assistant_message

    except Exception as e:
        logger.error(f"Error processing chat message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def update_chat_document(
    user_id: Optional[str],
    chat_id: str,
    session_id: str,
    message: str,
    redis_client: Redis,
):
    """Update or create chat document in Firestore."""
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
        if user_id:
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
):
    """Create or update session document in Firestore."""
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


async def generate_default_chat_name(message: str) -> str:
    """Generate a default name for a new chat."""
    title_prompt = f"Generate a short, catchy title (max 5 words) for a chat that starts with this message: '{message}'"
    messages = [HumanMessage(content=title_prompt)]
    response = await llm.ainvoke(messages)
    return response.content.strip().strip("\"'")


@lru_cache(maxsize=100)
async def get_cached_chat_name(chat_id: str, redis_client: Redis) -> str:
    """Get chat name from cache or Firestore."""
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


async def get_chat_messages(
    chat_id: str, limit: Optional[int] = None
) -> Dict[str, Any]:
    """Get all messages for a chat."""
    logger.debug(f"Fetching chat messages for chat_id: {chat_id}")
    try:
        chat_history = FirestoreChatMessageHistory(chat_id)
        messages = await chat_history.get_messages(limit)
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


async def retry_message(
    user_id: str,
    message_id: str,
    background_tasks: BackgroundTasks,
    redis: Redis,
) -> AsyncGenerator[ChatMessage, None]:
    """Retry processing a specific message."""
    chat_sessions_ref = db.collection("chat_sessions")
    query = chat_sessions_ref.where("user_id", "==", user_id)
    sessions = await query.get()

    for session in sessions:
        messages_ref = session.reference.collection("messages")
        message_doc = await messages_ref.document(message_id).get()

        if message_doc.exists:
            previous_messages = (
                await messages_ref.where("timestamp", "<", message_doc.get("timestamp"))
                .order_by("timestamp", direction=firestore.Query.DESCENDING)
                .limit(1)
                .get()
            )

            for prev_message in previous_messages:
                if prev_message.get("role") == "human":
                    user_message = prev_message.to_dict()
                    async for response in process_chat_message(
                        user_id=user_id,
                        chat_id=session.id,
                        session_id=session.id,
                        client_ip="",
                        message=user_message["content"],
                        user_agent="",
                        voice_type="normal",
                        background_tasks=background_tasks,
                        redis=redis,
                    ):
                        yield response

    raise HTTPException(
        status_code=404, detail="Message not found or not eligible for retry"
    )
