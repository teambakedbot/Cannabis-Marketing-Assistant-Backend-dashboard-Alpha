from datetime import datetime, timedelta
from firebase_admin import auth
from ..utils.firebase_utils import db
from ..tools.tools import llm
import os
import logging
from typing import List
from redis.asyncio import Redis
from google.cloud import firestore
from google.protobuf.timestamp_pb2 import Timestamp
import json
from ..utils.redis_config import FirestoreEncoder
from ..config.config import logger
from ..models.schemas import ChatMessage
from langchain_core.messages import HumanMessage


# Helper functions for managing conversation context
async def get_conversation_context(
    chat_id: str, redis_client: Redis, max_context_length: int = 5
):
    # Try to get context from Redis
    context = await redis_client.get(f"context:{chat_id}")
    if context == "d":
        logger.debug("Retrieved conversation context from Redis: %s", context)
        return json.loads(context)[-max_context_length:]
    else:
        chat_ref = db.collection("chats").document(chat_id)
        chat_doc = chat_ref.get()

        if chat_doc.exists:
            messages_collection = chat_ref.collection("messages")
            messages = (
                messages_collection.order_by(
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
            logger.debug(f"Cached conversation context in Redis for chat {chat_id}")
            return context

        logger.debug(f"No chat found for chat {chat_id}")
        return []


def update_conversation_context(session_ref, context: List[dict]):
    logger.debug("Updating conversation context in Firestore: %s", context)
    session_ref.set({"context": context})
    logger.debug("Conversation context updated successfully")


async def summarize_context(context: List[dict], redis_client: Redis):
    summary_prompt = "Summarize the following conversation context briefly: "
    full_context = " ".join([f"{msg['role']}: {msg['content']}" for msg in context])
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
        {
            "message_id": "summary",
            "role": "system",
            "content": summary.content,
            "timestamp": datetime.now(),
        }
    ]
    # Cache the summary with an expiration time
    await redis_client.set(
        context_key, json.dumps(summary_content, cls=FirestoreEncoder), ex=3600
    )

    return summary_content


# Background task to clean up old sessions and logout users
def clean_up_old_sessions():
    logger.debug("Starting cleanup of old sessions")
    cutoff_date = datetime.utcnow() - timedelta(days=30)
    sessions_ref = db.collection("sessions")
    old_sessions = sessions_ref.where("timestamp", "<", cutoff_date).stream()

    for session in old_sessions:
        session_data = session.to_dict()
        if session_data.get("user_id"):  # If user_id exists, revoke tokens
            try:
                auth.revoke_refresh_tokens(session_data["user_id"])
                logger.debug(
                    f"Revoked Firebase token for user_id: {session_data['user_id']}"
                )
            except Exception as e:
                logger.error(
                    f"Error revoking Firebase token for user_id: {session_data['user_id']} - {str(e)}"
                )

        session.reference.delete()
        logger.debug(f"Deleted old session: {session.id}")


# Function to merge unauthenticated session into authenticated user chat history
def merge_unauthenticated_session_to_user(session_id: str, user_id: str) -> str:
    logger.debug(f"Ending session: {session_id}")
    session_ref = db.collection("sessions").document(session_id)
    session_doc = session_ref.get()
    logger.debug(
        f"Checking if session document exists for merging: {session_doc.exists}"
    )
    if session_doc.exists:
        session_data = session_doc.to_dict()
        context = session_data.get("context", [])
        if context:  # Only merge if there is context to save
            chat_id = os.urandom(16).hex()  # Generate a new chat ID
            logger.debug("Created new chat_id: %s for user_id: %s", chat_id, user_id)
            chat_ref = (
                db.collection("user_chats")
                .document(user_id)
                .collection("chats")
                .document(chat_id)
            )
            chat_entry = {
                "context": [
                    {
                        "role": msg["role"],
                        "content": msg["content"],
                        "session_id": session_id,
                    }
                    for msg in context
                ],
            }
            logger.debug(
                "Chat entry to be saved: %s",
                chat_entry,
            )
            logger.debug("Saving merged chat entry to Firestore")
            chat_ref.set(chat_entry, merge=True)
            logger.debug("Merged chat entry saved successfully")

            # Once merged, delete the unauthenticated session to avoid duplication
            session_ref.delete()
            logger.debug(f"Merged and deleted unauthenticated session: {session_id}")

            return chat_id  # Return the new chat_id
    logger.debug(
        "No context found in session_id: %s for user_id: %s", session_id, user_id
    )
    return None  # No chat_id if nothing was merged


# Function to end session and calculate session length
def end_session(session_id: str):
    session_ref = db.collection("sessions").document(session_id)
    session_doc = session_ref.get()
    if session_doc.exists:
        session_data = session_doc.to_dict()
        start_time = session_data.get("start_time")
        if start_time:
            end_time = datetime.utcnow()
            session_length = end_time - start_time
            logger.debug(
                f"Session start time found: {start_time}, calculating session length"
            )
            session_ref.update(
                {
                    "end_time": end_time,
                    "session_length": session_length.total_seconds()
                    // 60,  # Store length in minutes
                }
            )
            logger.debug(
                f"Session {session_id} ended. Length: {session_length.total_seconds() // 60} minutes."
            )


async def generate_default_chat_name(message: str) -> str:
    title_prompt = f"Generate a short, catchy title (max 5 words) for a chat that starts with this message: '{message}'"
    messages = [HumanMessage(content=title_prompt)]
    response = await llm.ainvoke(messages)
    return response.content.strip().strip("\"'")
