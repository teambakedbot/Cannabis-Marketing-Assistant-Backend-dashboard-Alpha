import os
import logging
from google.cloud import firestore
from google.protobuf.timestamp_pb2 import Timestamp
from fastapi import HTTPException
from .firebase_utils import db
from .tools import agent, agent_executor
from .utils import (
    summarize_context,
    generate_default_chat_name,
    clean_up_old_sessions,
    get_conversation_context,
)
from google.cloud import firestore
from .schemas import ChatRequest, ChatResponse, ChatMessage
from redis import Redis
from fastapi.background import BackgroundTasks
from typing import Optional, List, Dict
from datetime import datetime
import json
from .config import logger
from langchain.schema import HumanMessage, AIMessage
from langchain.schema.runnable import RunnableConfig
import asyncio
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from functools import lru_cache


class AsyncStreamingStdOutCallbackHandler(StreamingStdOutCallbackHandler):
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)


@lru_cache(maxsize=100)
def get_cached_chat_name(chat_id: str) -> str:
    chat_ref = db.collection("chats").document(chat_id)
    chat_doc = chat_ref.get()
    if chat_doc.exists:
        return chat_doc.to_dict().get("name", "Unnamed Chat")
    return "Unnamed Chat"


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
        # Add debugging information
        logger.debug(f"Redis client: {redis_client}")

        if redis_client is None:
            logger.error("Redis client is None")
            raise ValueError("Redis client is not initialized")

        if not chat_id:
            chat_id = os.urandom(16).hex()
            logger.debug(f"Generated new chat_id: {chat_id}")

        # Use a transaction for atomic operations
        @firestore.transactional
        def create_or_update_chat(transaction):
            chat_ref = db.collection("chats").document(chat_id)
            chat_doc = chat_ref.get(transaction=transaction)
            if not chat_doc.exists:
                default_name = generate_default_chat_name(message)
                chat_data = {
                    "chat_id": chat_id,
                    "user_ids": [user_id] if user_id else [],
                    "session_ids": [session_id] if not user_id else [],
                    "created_at": firestore.SERVER_TIMESTAMP,
                    "last_updated": firestore.SERVER_TIMESTAMP,
                    "name": default_name,
                }
                transaction.set(chat_ref, chat_data)
                logger.debug(f"New chat document created with chat_id: {chat_id}")
            else:
                chat_data = chat_doc.to_dict()
                updates = {}
                if user_id and user_id not in chat_data.get("user_ids", []):
                    updates["user_ids"] = firestore.ArrayUnion([user_id])
                if not user_id and session_id not in chat_data.get("session_ids", []):
                    updates["session_ids"] = firestore.ArrayUnion([session_id])
                if updates:
                    updates["last_updated"] = firestore.SERVER_TIMESTAMP
                    transaction.update(chat_ref, updates)
                    logger.debug(f"Chat document updated with chat_id: {chat_id}")

        # Execute the transaction
        transaction = db.transaction()
        create_or_update_chat(transaction)

        # Manage session
        session_ref = db.collection("sessions").document(session_id)
        session_doc = session_ref.get()
        if not session_doc.exists:
            session_data = {
                "session_id": session_id,
                "timestamp": firestore.SERVER_TIMESTAMP,
                "user_agent": user_agent,
                "ip_address": client_ip,
                "user_id": user_id if user_id else None,
            }
            session_ref.set(session_data, merge=True)
        elif user_id and user_id != session_doc.to_dict().get("user_id"):
            session_ref.update({"user_id": user_id})

        # Retrieve conversation context using Redis
        context_task = asyncio.create_task(
            get_conversation_context(session_id, redis_client, max_context_length=10)
        )

        # Wait for the context to be retrieved
        context = await context_task

        new_message = {
            "message_id": os.urandom(16).hex(),
            "user_id": user_id if user_id else None,
            "session_id": session_id,
            "role": "user",
            "content": message,
            "timestamp": firestore.SERVER_TIMESTAMP,
        }
        context.append(new_message)
        if len(context) > 10:
            context = await summarize_context(context, redis_client)
        else:
            context = context[-10:]

        chat_history = [
            ChatMessage(
                id=msg["message_id"],
                role=msg["role"],
                content=msg["content"],
                session_id=msg["session_id"],
                timestamp=(
                    datetime.now()
                    if msg["timestamp"] == firestore.SERVER_TIMESTAMP
                    else msg["timestamp"]
                ),
                is_from_user=(msg["role"] == "user"),
            )
            for msg in context
        ]
        # Define voice prompts
        voice_prompts = {
            "normal": "You are an AI-powered chatbot specialized in assisting cannabis marketers. Your name is BakedBot.",
            "pops": "You are a fatherly and upbeat AI assistant, ready to help with cannabis marketing. But you sound like Pops from the movie Friday, use his style of talk.",
            "smokey": "You are a laid-back and cool AI assistant, providing cannabis marketing insights. But sounds like Smokey from the movie Friday, use his style of talk.",
        }
        voice_prompt = voice_prompts.get(voice_type, voice_prompts["normal"])
        # Create the prompt to send to the AI agent
        new_prompt = (
            f"{voice_prompt} Instructions: {message}. Always OUTPUT in markdown."
        )

        # Convert chat history to the format expected by the agent
        langchain_history = [
            (
                HumanMessage(content=msg.content)
                if msg.is_from_user
                else AIMessage(content=msg.content)
            )
            for msg in chat_history
        ]

        # Use CallbackManager instead of AsyncCallbackManager
        callback_manager = CallbackManager([AsyncStreamingStdOutCallbackHandler()])

        # Create an async version of the agent executor
        async def async_agent_executor():
            config = RunnableConfig(callbacks=callback_manager)
            result = await agent_executor.ainvoke(
                {"input": new_prompt, "chat_history": langchain_history}, config=config
            )
            return result

        # Execute the agent
        agent_response = await async_agent_executor()

        # Extract the response text
        response_text = agent_response.get("output", "No response available.")
        if isinstance(response_text, dict):
            response_text = response_text.get("output", "No response available.")

        # Save the assistant's response in the chat history
        assistant_message = {
            "message_id": os.urandom(16).hex(),
            "user_id": None,
            "session_id": session_id,
            "role": "assistant",
            "content": response_text,
            "timestamp": firestore.SERVER_TIMESTAMP,
        }
        # Store the new user message and assistant response
        messages_ref = db.collection("chats").document(chat_id).collection("messages")
        batch = db.batch()
        batch.set(messages_ref.document(new_message["message_id"]), new_message)
        batch.set(
            messages_ref.document(assistant_message["message_id"]), assistant_message
        )
        batch.update(
            db.collection("chats").document(chat_id),
            {"last_updated": firestore.SERVER_TIMESTAMP},
        )
        batch.commit()

        # Schedule the cleanup task for old sessions
        background_tasks.add_task(clean_up_old_sessions)
        # Return the chat response along with the chat_id
        return ChatResponse(response=response_text, chat_id=chat_id)
    except Exception as e:
        logger.error(f"Error occurred in process_chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def rename_chat(chat_id: str, new_name: str, user_id: str):
    try:
        if not new_name:
            raise HTTPException(status_code=400, detail="New name must be provided")

        # Reference the chat document
        chat_ref = db.collection("chats").document(chat_id)
        chat_doc = chat_ref.get()
        if not chat_doc.exists:
            raise HTTPException(status_code=404, detail="Chat not found")

        chat_data = chat_doc.to_dict()
        if user_id not in chat_data.get("user_ids", []):
            raise HTTPException(
                status_code=403, detail="User not authorized to rename this chat"
            )

        # Update the chat name
        chat_ref.update({"name": new_name, "last_updated": firestore.SERVER_TIMESTAMP})
        logger.debug(f"Chat {chat_id} renamed to {new_name} by user {user_id}")

        return {"message": "Chat renamed successfully"}

    except Exception as e:
        logger.error(f"Error occurred while renaming chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def archive_chat(chat_id: str, user_id: str):
    try:
        # Reference the chat document
        chat_ref = db.collection("chats").document(chat_id)
        chat_doc = chat_ref.get()
        if not chat_doc.exists:
            raise HTTPException(status_code=404, detail="Chat not found")

        chat_data = chat_doc.to_dict()
        if user_id not in chat_data.get("user_ids", []):
            raise HTTPException(
                status_code=403, detail="User not authorized to archive this chat"
            )

        # Update the 'archived' field
        chat_ref.update({"archived": True, "last_updated": firestore.SERVER_TIMESTAMP})
        logger.debug(f"Chat {chat_id} archived by user {user_id}")

        return {"message": "Chat archived successfully"}

    except Exception as e:
        logger.error(f"Error occurred while archiving chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def delete_chat(chat_id: str, user_id: str):
    try:
        # Reference the chat document
        chat_ref = db.collection("chats").document(chat_id)
        chat_doc = chat_ref.get()
        if not chat_doc.exists:
            raise HTTPException(status_code=404, detail="Chat not found")

        chat_data = chat_doc.to_dict()
        if user_id not in chat_data.get("user_ids", []):
            raise HTTPException(
                status_code=403, detail="User not authorized to delete this chat"
            )

        # Delete the chat document and its messages
        messages_ref = chat_ref.collection("messages")
        batch = db.batch()
        messages = messages_ref.stream()
        for msg in messages:
            batch.delete(msg.reference)
        batch.delete(chat_ref)
        batch.commit()

        logger.debug(f"Chat {chat_id} and its messages deleted by user {user_id}")

        return {"message": "Chat deleted successfully"}

    except Exception as e:
        logger.error(f"Error occurred while deleting chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_chat_messages(chat_id: str):
    logger.debug("Fetching chat messages for chat_id: %s", chat_id)
    try:
        # Reference the chat document in the chats collection
        chat_ref = db.collection("chats").document(chat_id)
        chat_doc = chat_ref.get()
        if not chat_doc.exists:
            raise HTTPException(status_code=404, detail="Chat not found")

        # Retrieve messages from the chat
        messages_ref = chat_ref.collection("messages")
        messages = messages_ref.order_by("timestamp").stream()
        chat_history = [msg.to_dict() for msg in messages]

        logger.debug("Chat messages retrieved for chat_id: %s", chat_id)
        return {"chat_id": chat_id, "messages": chat_history}

    except Exception as e:
        logger.error("Error occurred while fetching chat messages: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


async def record_feedback(user_id: str, message_id: str, feedback_type: str):
    """
    Record user feedback for a specific message.
    """
    feedback_ref = db.collection("feedback").document()
    feedback_data = {
        "user_id": user_id,
        "message_id": message_id,
        "feedback_type": feedback_type,
        "timestamp": firestore.SERVER_TIMESTAMP,
    }
    feedback_ref.set(feedback_data)
    return {"status": "success", "feedback_id": feedback_ref.id}


async def retry_message(
    user_id: str, message_id: str, background_tasks: BackgroundTasks, redis: Redis
):
    """
    Retry a specific message in the chat history.
    """
    # Get the chat session containing the message
    chat_sessions_ref = db.collection("chat_sessions")
    query = chat_sessions_ref.where("user_id", "==", user_id)
    sessions = query.stream()

    for session in sessions:
        messages_ref = session.reference.collection("messages")
        message_doc = messages_ref.document(message_id).get()

        if message_doc.exists:
            # Found the message, now get the previous user message
            previous_messages = (
                messages_ref.where("timestamp", "<", message_doc.get("timestamp"))
                .order_by("timestamp", direction=firestore.Query.DESCENDING)
                .limit(1)
                .stream()
            )

            for prev_message in previous_messages:
                if prev_message.get("is_from_user"):
                    # Process the previous user message again
                    user_message = prev_message.to_dict()
                    response = await process_chat_message(
                        user_id=user_id,
                        chat_id=session.id,
                        session_id=session.id,
                        client_ip="",  # You may want to store and retrieve this information
                        message=user_message["content"],
                        user_agent="",  # You may want to store and retrieve this information
                        voice_type="normal",  # You may want to store and retrieve this information
                        background_tasks=background_tasks,
                        redis=redis,
                    )
                    return response

    raise HTTPException(
        status_code=404, detail="Message not found or not eligible for retry"
    )
