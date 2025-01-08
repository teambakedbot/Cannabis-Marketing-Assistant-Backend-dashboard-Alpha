from typing import Optional, Any, TypeVar
from redis import Redis
from redis.exceptions import RedisError, ConnectionError
from functools import wraps
import asyncio
from ..config.config import logger, settings

T = TypeVar("T")


class RedisWrapper:
    def __init__(self, url: str, max_retries: int = 3, retry_delay: float = 1.0):
        self.url = url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client: Optional[Redis] = None

    @property
    def client(self) -> Redis:
        if self._client is None:
            try:
                self._client = Redis.from_url(
                    self.url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_timeout=5.0,
                    socket_connect_timeout=5.0,
                )
                # Test connection
                self._client.ping()
            except (RedisError, ConnectionError) as e:
                logger.error(f"Failed to initialize Redis connection: {e}")
                raise
        return self._client

    async def get_with_retry(self, key: str, default: Any = None) -> Any:
        """Get value from Redis with retry logic."""
        for attempt in range(self.max_retries):
            try:
                value = self.client.get(key)
                return value if value is not None else default
            except (RedisError, ConnectionError) as e:
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"Failed to get key {key} from Redis after {self.max_retries} attempts: {e}"
                    )
                    return default
                await asyncio.sleep(self.retry_delay * (attempt + 1))
                self._client = None  # Force reconnection

    async def set_with_retry(
        self, key: str, value: Any, expiration: Optional[int] = None
    ) -> bool:
        """Set value in Redis with retry logic."""
        for attempt in range(self.max_retries):
            try:
                if expiration:
                    return self.client.setex(key, expiration, value)
                return self.client.set(key, value)
            except (RedisError, ConnectionError) as e:
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"Failed to set key {key} in Redis after {self.max_retries} attempts: {e}"
                    )
                    return False
                await asyncio.sleep(self.retry_delay * (attempt + 1))
                self._client = None  # Force reconnection

    async def delete_with_retry(self, key: str) -> bool:
        """Delete key from Redis with retry logic."""
        for attempt in range(self.max_retries):
            try:
                return bool(self.client.delete(key))
            except (RedisError, ConnectionError) as e:
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"Failed to delete key {key} from Redis after {self.max_retries} attempts: {e}"
                    )
                    return False
                await asyncio.sleep(self.retry_delay * (attempt + 1))
                self._client = None  # Force reconnection


def with_redis_retry(max_retries: int = 3, retry_delay: float = 1.0):
    """Decorator for Redis operations with retry logic."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (RedisError, ConnectionError) as e:
                    last_error = e
                    if attempt == max_retries - 1:
                        logger.error(
                            f"Redis operation failed after {max_retries} attempts: {e}"
                        )
                        raise
                    await asyncio.sleep(retry_delay * (attempt + 1))
            raise last_error

        return wrapper

    return decorator


# Create a global Redis wrapper instance
redis_wrapper = RedisWrapper(settings.REDISCLOUD_URL)
