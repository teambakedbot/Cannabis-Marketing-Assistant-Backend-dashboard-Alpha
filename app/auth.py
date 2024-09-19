from fastapi import Header, Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
from .config import settings
from passlib.context import CryptContext

security = HTTPBearer()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password, hashed_password):
    """
    Verify a plain password against the hashed password.
    """
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password):
    """
    Hash a plain password.
    """
    return pwd_context.hash(password)


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )
    return encoded_jwt


def decode_token(token: str):
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def auth_middleware(request: Request, call_next):
    if request.url.path in ["/api/v1/login", "/api/v1/register", "/health", "/"]:
        return await call_next(request)

    return await call_next(request)

    # credentials: HTTPAuthorizationCredentials = await security(request)
    # if not credentials:
    #     raise HTTPException(
    #         status_code=401, detail="Invalid authentication credentials"
    #     )
    # print("$$$", credentials)

    # token = credentials.credentials
    # payload = decode_token(token)
    # user_id = payload.get("sub")
    # if user_id is None:
    #     raise HTTPException(status_code=401, detail="Invalid token")

    # user = get_user(request.state.db, user_id)
    # if user is None:
    #     raise HTTPException(status_code=401, detail="User not found")

    # request.state.user = user
    # return await call_next(request)
