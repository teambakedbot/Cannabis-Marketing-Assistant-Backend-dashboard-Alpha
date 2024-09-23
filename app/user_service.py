from .firebase_utils import db, verify_firebase_token
from fastapi import HTTPException, Depends
import logging
from redis.asyncio import Redis
from .redis_config import get_redis
import json
from .redis_config import FirestoreEncoder
from .config import logger


async def get_user_chats(
    user_id: str, redis_client: Redis, page: int = 1, page_size: int = 20
):
    offset = (page - 1) * page_size
    # Implement caching if necessary
    cache_key = f"user_chats:{user_id}:page:{page}:size:{page_size}"
    cached_chats = await redis_client.get(cache_key)
    if cached_chats:
        logger.debug(
            f"Retrieved user chats from Redis for user_id: {user_id}, page: {page}"
        )
        return json.loads(cached_chats)

    chats_ref = (
        db.collection("chats")
        .where("user_ids", "array_contains", user_id)
        .order_by("last_updated", "DESCENDING")
        .offset(offset)
        .limit(page_size)
    )
    chats = chats_ref.stream()
    chat_list = [{"chat_id": chat.id, **chat.to_dict()} for chat in chats]

    # Cache the result
    await redis_client.set(
        cache_key, json.dumps(chat_list, cls=FirestoreEncoder), ex=300
    )  # Cache for 5 minutes
    logger.debug(f"Cached user chats in Redis for user_id: {user_id}, page: {page}")

    return {"user_id": user_id, "chats": chat_list}
