from ..utils.firebase_utils import auth, verify_firebase_token, db as firestore_db
from fastapi import HTTPException, Depends, Header
from fastapi.security import APIKeyHeader
from ..models.schemas import User
from typing import Optional, Dict, Any
from ..config.config import logger
import json
from redis.asyncio import Redis
from ..utils.redis_config import get_redis, FirestoreEncoder

api_key_multiple = APIKeyHeader(name="Authorization", auto_error=False)

TOKEN_PREFIX = "token:"
TOKEN_EXPIRATION = 3600  # 1 hour


async def logout(user_id: str) -> Dict[str, str]:
    """Revokes refresh tokens for a user."""
    try:
        auth.revoke_refresh_tokens(user_id)
        logger.debug(f"Revoked tokens for user_id: {user_id}")
        return {"message": "Successfully logged out"}
    except Exception as e:
        logger.error("Error occurred during logout: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


async def _get_decoded_token(token: str, redis: Redis) -> Dict[str, Any]:
    """Helper function to get the decoded token from cache or verify it."""
    cached_token = await redis.get(f"{TOKEN_PREFIX}{token}")

    if cached_token:
        return json.loads(cached_token)

    decoded_token = await verify_firebase_token(token)
    await redis.set(
        f"{TOKEN_PREFIX}{token}",
        json.dumps(decoded_token, cls=FirestoreEncoder),
        ex=TOKEN_EXPIRATION,
    )
    return decoded_token


async def get_current_user_optional(
    authorization: str | None = Depends(api_key_multiple),
    redis: Redis = Depends(get_redis),
) -> Optional[User]:
    """Get the current user if authorization is provided."""
    if authorization:
        try:
            token = authorization.split(" ")[1]
            decoded_token = await _get_decoded_token(token, redis)
            user_id = decoded_token.get("uid")
            user = await firestore_db.collection("users").document(user_id).get()
            logger.debug("User authenticated with user_id: %s", user_id)
            return user
        except IndexError:
            return None
        except Exception as e:
            logger.error(f"Error retrieving user: {e}")
            return None
    return None


async def get_firebase_user(
    authorization: str = Depends(api_key_multiple),
    redis: Redis = Depends(get_redis),
) -> User:
    """Get the current user and raise an error if not found."""
    if authorization:
        try:
            token = authorization.split(" ")[1]
            decoded_token = await _get_decoded_token(token, redis)
            user_id = decoded_token.get("uid")
            user = await firestore_db.collection("users").document(user_id).get()
            logger.debug("User authenticated with user_id: %s", user_id)
            return user
        except IndexError:
            raise HTTPException(
                status_code=400, detail="Invalid authorization header format"
            )
        except Exception as e:
            logger.error(f"Error retrieving user: {e}")
            raise HTTPException(
                status_code=422,
                detail={
                    "Access denied: Your session has expired or the token is invalid. Please log in again."
                },
            )
    else:
        raise HTTPException(
            status_code=401, detail="Authorization header missing or invalid"
        )
