from .firebase_utils import db, verify_firebase_token
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)


async def get_user_chats(user_id):
    try:
        # Fetch all chats for the authenticated user
        chats_ref = (
            db.collection("chats")
            .where("user_ids", "array_contains", user_id)
            .order_by("last_updated", "DESCENDING")
        )
        chats = chats_ref.stream()
        chat_list = [{"chat_id": chat.id, **chat.to_dict()} for chat in chats]

        logger.debug("User chats retrieved for user_id: %s", user_id)
        return {"user_id": user_id, "chats": chat_list}

    except Exception as e:
        logger.error("Error occurred while fetching user chats: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
