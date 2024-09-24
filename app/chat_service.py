import os
import logging
from google.cloud import firestore
from google.protobuf.timestamp_pb2 import Timestamp
from fastapi import HTTPException
from .firebase_utils import db
from .tools import agent, ChatMessage
from .utils import (
    summarize_context,
    generate_default_chat_name,
    clean_up_old_sessions,
    get_conversation_context,
)
from google.cloud import firestore
from .schemas import ChatRequest, ChatResponse
from redis import Redis
from fastapi.background import BackgroundTasks
from typing import Optional
from datetime import datetime
import json
from .config import logger


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
        # Ensure chat_id is generated if not provided
        if not chat_id:
            chat_id = os.urandom(16).hex()
            logger.debug(f"Generated new chat_id: {chat_id}")
        # Reference the chat document in the chats collection
        chat_ref = db.collection("chats").document(chat_id)
        # Handle authentication

        # Set up chat document if it doesn't exist
        chat_doc = chat_ref.get()
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
            chat_ref.set(chat_data)
            logger.debug(f"New chat document created with chat_id: {chat_id}")
        else:
            chat_data = chat_doc.to_dict()
            if user_id and user_id not in chat_data.get("user_ids", []):
                chat_ref.update({"user_ids": firestore.ArrayUnion([user_id])})
            if not user_id and session_id not in chat_data.get("session_ids", []):
                chat_ref.update({"session_ids": firestore.ArrayUnion([session_id])})
            logger.debug(f"Chat document updated with chat_id: {chat_id}")
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
        context = await get_conversation_context(
            session_id, redis_client, max_context_length=10
        )
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
        # Convert context to ChatMessage objects
        chat_history = [
            ChatMessage(role=msg["role"], content=msg["content"]) for msg in context
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
        agent_response = agent.chat(message=new_prompt, chat_history=chat_history)
        response_text = (
            agent_response.response if agent_response else "No response available."
        )
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
        messages_ref.document(new_message["message_id"]).set(new_message)
        messages_ref.document(assistant_message["message_id"]).set(assistant_message)
        # Update the last_updated timestamp on the chat document
        chat_ref.update({"last_updated": firestore.SERVER_TIMESTAMP})
        # Schedule the cleanup task for old sessions
        background_tasks.add_task(clean_up_old_sessions)
        # Return the chat response along with the chat_id
        return ChatResponse(response=response_text, chat_id=chat_id)
    except Exception as e:
        logger.error("Error occurred in process_chat: %s", e)
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
        messages = messages_ref.stream()
        for msg in messages:
            msg.reference.delete()
        chat_ref.delete()

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
