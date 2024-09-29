import os
from redis.asyncio import Redis
import json
from google.cloud import firestore
from google.protobuf.timestamp_pb2 import Timestamp
from google.api_core.datetime_helpers import DatetimeWithNanoseconds
from datetime import datetime
from ..config.config import settings
from ..models.schemas import ChatMessage

redis_client: Redis = None


class FirestoreEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (Timestamp)):
            return str(obj)
        if isinstance(obj, DatetimeWithNanoseconds):
            return obj.isoformat()
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        if isinstance(obj, ChatMessage):
            return obj.__dict__
        return super().default(obj)


async def init_redis():
    global redis_client
    redis_url = settings.REDISCLOUD_URL
    redis_client = await Redis.from_url(
        redis_url, encoding="utf-8", decode_responses=True
    )


async def close_redis():
    if redis_client:
        await redis_client.close()


def get_redis() -> Redis:
    if redis_client is None:
        raise RuntimeError("Redis client is not initialized")
    return redis_client
