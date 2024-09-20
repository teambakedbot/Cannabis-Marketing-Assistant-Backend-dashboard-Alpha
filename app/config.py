from pydantic_settings import BaseSettings
import os
from typing import List


class Settings(BaseSettings):
    SECRET_KEY: str = os.getenv("SECRET_KEY", "1234567890")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES   ", 30)
    DEBUG: bool = os.getenv("DEBUG", True)
    WORKERS: int = 1
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv(
        "CELERY_RESULT_BACKEND", "redis://localhost:6379/0"
    )
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "1234567890")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "1234567890")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "1234567890")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    FIREBASE_CREDENTIALS: str = os.getenv("FIREBASE_CREDENTIALS", "1234567890")

    class Config:
        env_file = ".env"
        extra = "allow"  # Change this to allow extra fields


settings = Settings()
