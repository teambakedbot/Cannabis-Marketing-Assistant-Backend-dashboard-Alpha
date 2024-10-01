import os
from google.cloud import firestore
from fastapi import HTTPException
from ..utils.firebase_utils import db
from ..tools.tools import agent_executor
from ..utils.sessions import (
    clean_up_old_sessions,
)
from google.cloud import firestore
from ..models.schemas import (
    ChatMessage,
)
from redis import Redis
from fastapi.background import BackgroundTasks
from typing import Optional, List
from datetime import datetime, timedelta
from ..config.config import logger
from langchain.schema import HumanMessage, AIMessage
from langchain.schema.runnable import RunnableConfig
import asyncio
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from functools import lru_cache

from ..tools.tools import llm
import os
from typing import List
from redis.asyncio import Redis
from google.cloud import firestore
import json
from ..utils.redis_config import FirestoreEncoder
from ..config.config import logger
from ..models.schemas import ChatMessage
from langchain_core.messages import HumanMessage


class AsyncStreamingStdOutCallbackHandler(StreamingStdOutCallbackHandler):
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)


async def generate_default_chat_name(message: str) -> str:
    title_prompt = f"Generate a short, catchy title (max 5 words) for a chat that starts with this message: '{message}'"
    messages = [HumanMessage(content=title_prompt)]
    response = await llm.ainvoke(messages)
    return response.content.strip().strip("\"'")


@lru_cache(maxsize=100)
async def get_cached_chat_name(chat_id: str, redis_client: Redis) -> str:
    # Try to get from Redis first
    cached_name = await redis_client.get(f"chat_name:{chat_id}")
    if cached_name:
        return cached_name.decode("utf-8")

    # If not in Redis, fetch from Firestore
    chat_ref = db.collection("chats").document(chat_id)
    chat_doc = await chat_ref.get()
    if chat_doc.exists:
        name = chat_doc.to_dict().get("name", "Unnamed Chat")
    else:
        name = "Unnamed Chat"

    # Cache in Redis for future use
    await redis_client.set(f"chat_name:{chat_id}", name, ex=3600)  # Cache for 1 hour

    return name


# TODO: clear cache when renaming chat
async def rename_chat(chat_id: str, new_name: str, user_id: str):
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

    except Exception as e:
        logger.error(f"Error occurred while renaming chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def archive_chat(chat_id: str, user_id: str):
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


async def delete_chat(chat_id: str, user_id: str):
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


async def get_chat_messages(chat_id: str):
    logger.debug("Fetching chat messages for chat_id: %s", chat_id)
    try:
        chat_ref = db.collection("chats").document(chat_id)
        chat_doc = await chat_ref.get()
        if not chat_doc.exists:
            raise HTTPException(status_code=404, detail="Chat not found")

        messages_ref = chat_ref.collection("messages")
        messages = await messages_ref.order_by("timestamp").get()
        chat_history = [msg.to_dict() for msg in messages]

        logger.debug("Chat messages retrieved for chat_id: %s", chat_id)
        return {"chat_id": chat_id, "messages": chat_history}

    except Exception as e:
        logger.error("Error occurred while fetching chat messages: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


async def record_feedback(user_id: str, message_id: str, feedback_type: str):
    feedback_ref = db.collection("feedback").document()
    feedback_data = {
        "user_id": user_id,
        "message_id": message_id,
        "feedback_type": feedback_type,
        "timestamp": firestore.SERVER_TIMESTAMP,
    }
    await feedback_ref.set(feedback_data)
    return {"status": "success", "feedback_id": feedback_ref.id}


# TODO: Refactor this to fit the new ProcessChatMessage function and manage history better
async def retry_message(
    user_id: str, message_id: str, background_tasks: BackgroundTasks, redis: Redis
):
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
                if prev_message.get("role") == "user":
                    user_message = prev_message.to_dict()
                    response = await process_chat_message(
                        user_id=user_id,
                        chat_id=session.id,
                        session_id=session.id,
                        client_ip="",
                        message=user_message["content"],
                        user_agent="",
                        voice_type="normal",
                        background_tasks=background_tasks,
                        redis=redis,
                    )
                    return response

    raise HTTPException(
        status_code=404, detail="Message not found or not eligible for retry"
    )


async def process_chat_message(
    user_id: Optional[str],
    chat_id: Optional[str],
    session_id: str,
    client_ip: str,
    message: str,
    user_agent: str,
    voice_type: str,
    background_tasks: BackgroundTasks,
    redis_client: Redis,
):
    try:
        if not chat_id:
            chat_id = os.urandom(16).hex()
            logger.debug(f"Generated new chat_id: {chat_id}")

        await update_chat_document(user_id, chat_id, session_id, message, redis_client)
        await manage_session(session_id, user_agent, client_ip, user_id)

        chat_history_task = asyncio.create_task(
            get_conversation_history(chat_id, redis_client, max_context_length=50)
        )

        chat_history = await chat_history_task

        langchain_history = []
        for msg in chat_history:
            if hasattr(msg, "to_dict"):
                msg_dict = msg.to_dict()
            else:
                msg_dict = msg

            if msg_dict.get("role") == "user":
                langchain_history.append(
                    HumanMessage(content=msg_dict.get("content", ""))
                )
            elif msg_dict.get("role") == "assistant":
                langchain_history.append(
                    AIMessage(
                        content=msg_dict.get("content", ""), data=msg_dict.get("data")
                    )
                )

        user_message = ChatMessage(
            chat_id=chat_id,
            message_id=os.urandom(16).hex(),
            user_id=user_id if user_id else None,
            session_id=session_id,
            role="user",
            content=message,
            timestamp=datetime.now(),
        )
        langchain_history.append(HumanMessage(content=message))

        if len(langchain_history) > 10:
            langchain_history = await summarize_context(langchain_history, redis_client)

        voice_prompts = {
            "normal": "You are an AI-powered chatbot specialized in assisting cannabis marketers. Your name is Smokey.",
            "pops": "You are a fatherly and upbeat AI assistant, ready to help with cannabis marketing. But you sound like Pops from the movie Friday, use his style of talk.",
            "smokey": "You are a laid-back and cool AI assistant, providing cannabis marketing insights. But sounds like Smokey from the movie Friday, use his style of talk.",
        }
        voice_prompt = voice_prompts.get(voice_type, voice_prompts["normal"])
        new_prompt = (
            f"{voice_prompt} Instructions: {message}. Always OUTPUT in markdown."
        )

        callback_manager = CallbackManager([AsyncStreamingStdOutCallbackHandler()])

        async def async_agent_executor():
            config = RunnableConfig(callbacks=callback_manager)
            result = await agent_executor.ainvoke(
                {"input": new_prompt, "chat_history": langchain_history},
                config=config,
            )
            return result

        agent_response = await async_agent_executor()
        response_text = ""
        data = None

        if isinstance(agent_response, dict):
            output = agent_response.get("output", "")
            if isinstance(output, tuple):
                response_text, product_data = output
                if product_data:
                    data = {"products": json.loads(product_data).get("products", [])}
            else:
                response_text = output
        else:
            response_text = str(agent_response)

        assistant_message = ChatMessage(
            message_id=os.urandom(16).hex(),
            user_id=None,
            session_id=session_id,
            role="assistant",
            content=response_text,
            chat_id=chat_id,
            data=data,
            timestamp=datetime.now() + timedelta(milliseconds=1),
        )

        await store_messages(chat_id, user_message, assistant_message)

        background_tasks.add_task(clean_up_old_sessions)
        return assistant_message
    except Exception as e:
        logger.error(f"Error occurred in process_chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# add users and session ids to the chat document
async def update_chat_document(
    user_id: Optional[str],
    chat_id: str,
    session_id: str,
    message: str,
    redis_client: Redis,
):
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


# add users and session ids to the session document
async def manage_session(
    session_id: str, user_agent: str, client_ip: str, user_id: Optional[str]
):
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


# store messages in the chat document
async def store_messages(
    chat_id: str, new_message: ChatMessage, assistant_message: ChatMessage
):
    messages_ref = db.collection("chats").document(chat_id).collection("messages")
    chat_ref = db.collection("chats").document(chat_id)

    batch = db.batch()
    batch.set(messages_ref.document(new_message.message_id), new_message.dict())
    batch.set(
        messages_ref.document(assistant_message.message_id), assistant_message.dict()
    )
    batch.update(chat_ref, {"updated_at": firestore.SERVER_TIMESTAMP})
    await batch.commit()


async def summarize_context(context: List[ChatMessage], redis_client: Redis):
    summary_prompt = "Summarize the following conversation context briefly: "

    full_context = " ".join([f"{msg.role}: {msg.content}" for msg in context])
    summarization_input = summary_prompt + full_context

    # Generate a unique cache key based on the context
    context_key = f"summary:{hash(summarization_input)}"
    cached_summary = await redis_client.get(context_key)
    if cached_summary:
        logger.debug("Retrieved summary from Redis: %s", cached_summary)
        return json.loads(cached_summary)

    summary = await llm.ainvoke(summarization_input)
    logger.debug("Context summarized: %s", summary.content)

    summary_content = [
        ChatMessage(
            chat_id=context[0].chat_id,
            message_id="summary",
            user_id=None,
            session_id=context[0].session_id,
            role="system",
            content=summary.content,
            timestamp=datetime.now(),
        )
    ]
    # Cache the summary with an expiration time
    await redis_client.set(
        context_key, json.dumps(summary_content, cls=FirestoreEncoder), ex=3600
    )

    return summary_content


async def get_conversation_history(
    chat_id: str, redis_client: Redis, max_context_length: int = 10
):
    context = await redis_client.get(f"context:{chat_id}")
    if context:
        return json.loads(context)[-max_context_length:]
    else:
        chat_ref = db.collection("chats").document(chat_id)
        messages_collection = chat_ref.collection("messages")
        messages = (
            await messages_collection.order_by(
                "timestamp", direction=firestore.Query.DESCENDING
            )
            .limit(max_context_length)
            .get()
        )

        context = [msg.to_dict() for msg in messages][::-1]

        await redis_client.set(
            f"context:{chat_id}",
            json.dumps(context, cls=FirestoreEncoder),
            ex=3600,
        )
        return context


async def update_conversation_context(session_ref, context: List[dict]):
    logger.debug("Updating conversation context in Firestore: %s", context)
    await session_ref.set({"context": context})
    logger.debug("Conversation context updated successfully")
