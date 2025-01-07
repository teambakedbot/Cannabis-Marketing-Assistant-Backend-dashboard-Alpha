import os
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from .utils.firebase_utils import initialize_firebase

from .utils.redis_config import init_redis, close_redis, get_redis
from .routes.routes import router
from .config.config import settings
from .core.exceptions.exceptions import CustomException
from redis.asyncio import Redis
import logging
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from .config.config import logger
from .routes import gemma_chat


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Smokey API",
    description="API for Smokey, an AI-powered cannabis product recommendation system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize Firebase
initialize_firebase()

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)

if settings.ENVIRONMENT == "production":
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)
    app.add_middleware(HTTPSRedirectMiddleware)

app.include_router(router)
app.include_router(gemma_chat.router, tags=["gemma"])


@app.on_event("startup")
async def startup_event():
    await init_redis()


@app.on_event("shutdown")
async def shutdown_event():
    await close_redis()


@app.get("/")
@limiter.limit("5/minute")
async def root(request: Request):
    return {"message": "Hello World"}
