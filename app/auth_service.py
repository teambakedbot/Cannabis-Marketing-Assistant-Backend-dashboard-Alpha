from .firebase_utils import auth, verify_firebase_token, db as firestore_db
from fastapi import HTTPException, Depends, Header
from fastapi.security import OAuth2PasswordBearer
import logging
from .models import User
from jose import JWTError, jwt
from typing import Optional
from . import models
from .config import settings

logger = logging.getLogger(__name__)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


async def logout(user_id: str):
    try:
        auth.revoke_refresh_tokens(user_id)
        logger.debug(f"Revoked tokens for user_id: {user_id}")
        return {"message": "Successfully logged out"}
    except Exception as e:
        logger.error("Error occurred during logout: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


async def get_current_user_optional(
    token: str = Depends(oauth2_scheme),
) -> Optional[dict]:
    if token is None:
        return None
    try:
        decoded_token = verify_firebase_token(token)
        user_id = decoded_token.get("uid")
        user = firestore_db.collection("users").document(user_id).get()
        return user.to_dict() if user.exists else None
    except Exception:
        return None


async def get_firebase_user(authorization: str = Header(None)):
    if authorization:
        try:
            token = authorization.split(" ")[1]
            decoded_token = verify_firebase_token(token)
            user_id = decoded_token.get("uid")
            user = firestore_db.collection("users").document(user_id).get()
            logger.debug("User authenticated with user_id: %s", user_id)
        except IndexError:
            raise HTTPException(
                status_code=400, detail="Invalid authorization header format"
            )
    else:
        raise HTTPException(
            status_code=401, detail="Authorization header missing or invalid"
        )
    return user
