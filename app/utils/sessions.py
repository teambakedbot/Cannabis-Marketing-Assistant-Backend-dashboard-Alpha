from datetime import datetime, timedelta
from firebase_admin import auth
from .firebase_utils import db
import os
from ..config.config import logger


# Helper functions for managing conversation context


# Background task to clean up old sessions and logout users
async def clean_up_old_sessions():
    logger.debug("Starting cleanup of old sessions")
    cutoff_date = datetime.utcnow() - timedelta(days=30)
    sessions_ref = db.collection("sessions")
    old_sessions = await sessions_ref.where("timestamp", "<", cutoff_date).get()

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

        await session.reference.delete()
        logger.debug(f"Deleted old session: {session.id}")


# Function to merge unauthenticated session into authenticated user chat history
async def merge_unauthenticated_session_to_user(session_id: str, user_id: str) -> str:
    logger.debug(f"Ending session: {session_id}")
    session_ref = db.collection("sessions").document(session_id)
    session_doc = await session_ref.get()
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
            await chat_ref.set(chat_entry, merge=True)
            logger.debug("Merged chat entry saved successfully")

            # Once merged, delete the unauthenticated session to avoid duplication
            await session_ref.delete()
            logger.debug(f"Merged and deleted unauthenticated session: {session_id}")

            return chat_id  # Return the new chat_id
    logger.debug(
        "No context found in session_id: %s for user_id: %s", session_id, user_id
    )
    return None  # No chat_id if nothing was merged


# Function to end session and calculate session length
async def end_session(session_id: str):
    session_ref = db.collection("sessions").document(session_id)
    session_doc = await session_ref.get()
    if session_doc.exists:
        session_data = session_doc.to_dict()
        start_time = session_data.get("start_time")
        if start_time:
            end_time = datetime.utcnow()
            session_length = end_time - start_time
            logger.debug(
                f"Session start time found: {start_time}, calculating session length"
            )
            await session_ref.update(
                {
                    "end_time": end_time,
                    "session_length": session_length.total_seconds()
                    // 60,  # Store length in minutes
                }
            )
            logger.debug(
                f"Session {session_id} ended. Length: {session_length.total_seconds() // 60} minutes."
            )
