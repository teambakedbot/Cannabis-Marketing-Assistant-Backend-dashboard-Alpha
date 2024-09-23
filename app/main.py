import os
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from .firebase_utils import initialize_firebase
from .redis_config import init_redis, close_redis, get_redis
from .routes import router
from .config import settings
from .exceptions import CustomException
from redis.asyncio import Redis

app = FastAPI(
    title="Smokey API",
    description="API for Smokey, an AI-powered cannabis product recommendation system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add SessionMiddleware
app.add_middleware(
    SessionMiddleware, secret_key=os.getenv("SESSION_SECRET_KEY", "your-secret-key")
)


@app.exception_handler(CustomException)
async def custom_exception_handler(request: Request, exc: CustomException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )


@app.on_event("startup")
async def startup_event():
    await init_redis()
    initialize_firebase()


@app.on_event("shutdown")
async def shutdown_event():
    await close_redis()


app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    return {"message": "Welcome to Smokey API. Visit /docs for the API documentation."}


@app.get("/health", tags=["Health Check"])
async def health_check(redis: Redis = Depends(get_redis)):
    try:
        await redis.ping()
        return {"status": "healthy", "redis": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "redis": "disconnected", "error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=settings.DEBUG,
        workers=settings.WORKERS,
    )
