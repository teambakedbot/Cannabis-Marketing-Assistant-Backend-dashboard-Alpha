import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware

from .database import engine, Base
from .routes import router
from .config import settings
from .auth import auth_middleware
from .exceptions import CustomException


Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Smokey API",
    description="API for Smokey, an AI-powered cannabis product recommendation system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(auth_middleware)

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


app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    return {"message": "Welcome to Smokey API. Visit /docs for the API documentation."}


@app.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=settings.DEBUG,
        workers=settings.WORKERS,
    )
