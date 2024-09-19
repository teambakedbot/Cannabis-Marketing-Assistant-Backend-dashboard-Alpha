from .firebase_utils import auth, verify_firebase_token, db as firestore_db
from fastapi import HTTPException, Depends, Header
from fastapi.security import OAuth2PasswordBearer
import logging
from .models import User
from .database import get_current_user, get_db
from jose import JWTError, jwt
from sqlalchemy.orm import Session
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


def get_current_active_user(current_user: User = Depends(get_current_user)):
    # Add logic to check if the user is active
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_current_user_optional(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> Optional[models.User]:
    if token is None:
        return None
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        user_id: int = payload.get("sub")
        if user_id is None:
            return None
    except JWTError:
        return None
    user = db.query(models.User).filter(models.User.id == user_id).first()
    return user


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
