import os
from redis.asyncio import Redis
import json
from google.cloud import firestore
from google.protobuf.timestamp_pb2 import Timestamp
from google.api_core.datetime_helpers import DatetimeWithNanoseconds
from datetime import datetime

redis_client = None


class FirestoreEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (Timestamp)):
            return str(obj)
        if isinstance(obj, DatetimeWithNanoseconds):
            return obj.isoformat()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


async def init_redis():
    global redis_client
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
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
