from datetime import datetime, timedelta
from firebase_admin import auth
from .firebase_utils import db
from .tools import llm, ChatMessage
import os
import logging
from typing import List
from redis.asyncio import Redis
from google.cloud import firestore
from google.protobuf.timestamp_pb2 import Timestamp
import json
from .redis_config import FirestoreEncoder

logger = logging.getLogger(__name__)


# Helper functions for managing conversation context
async def get_conversation_context(
    session_id: str, redis_client: Redis, max_context_length: int = 5
):
    # Try to get context from Redis
    context = await redis_client.get(f"context:{session_id}")
    if context:
        logger.debug("Retrieved conversation context from Redis: %s", context)
        return json.loads(context)
    else:
        # Fetch from Firestore and cache it
        session_ref = db.collection("sessions").document(session_id)
        session_doc = session_ref.get()
        if session_doc.exists:
            context = session_doc.to_dict().get("context", [])
            await redis_client.set(
                f"context:{session_id}",
                json.dumps(context, cls=FirestoreEncoder),
                ex=3600,
            )
            return context[-max_context_length:]
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

    summary = llm.chat([ChatMessage(role="system", content=summarization_input)])
    logger.debug("Context summarized: %s", summary.message.content)
    summary_content = [{"role": "system", "content": summary.message.content}]

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


def generate_default_chat_name(message: str) -> str:
    """Generate a default chat name using the LLM based on the provided message."""
    title_prompt = "Generate a short title for the following chat history: "
    title_input = title_prompt + message
    title_response = llm.chat(
        [
            ChatMessage(
                role="system",
                content=title_input,
            )
        ]
    )
    return title_response.message.content.strip()
