from ..utils.firebase_utils import db
from fastapi import HTTPException, Depends
import logging
from redis.asyncio import Redis
from ..utils.redis_config import get_redis
import json
from ..utils.redis_config import FirestoreEncoder
from ..config.config import logger


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
        return {"user_id": user_id, "chats": json.loads(cached_chats)}
    chats_ref = (
        db.collection("chats")
        .where("user_ids", "array_contains", user_id)
        .order_by("updated_at", "DESCENDING")
        .offset(offset)
        .limit(page_size)
    )
    chats = await chats_ref.get()
    chat_list = [{"chat_id": chat.id, **chat.to_dict()} for chat in chats]
    # Cache the result
    await redis_client.set(
        cache_key, json.dumps(chat_list, cls=FirestoreEncoder), ex=3
    )  # Cache for 5 minutes
    logger.debug(f"Cached user chats in Redis for user_id: {user_id}, page: {page}")

    return {"user_id": user_id, "chats": chat_list}
