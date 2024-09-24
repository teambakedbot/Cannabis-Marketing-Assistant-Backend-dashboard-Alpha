from pydantic_settings import BaseSettings
import logging
from logging.handlers import RotatingFileHandler
import os


class Settings(BaseSettings):
    SECRET_KEY: str = os.getenv("SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
    )
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    WORKERS: int = int(os.getenv("WORKERS", "1"))
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = os.getenv("AWS_REGION")
    FIREBASE_CREDENTIALS: str = os.getenv("FIREBASE_CREDENTIALS")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    CANNMENUS_API_KEY: str = os.getenv("CANNMENUS_API_KEY")
    IDEOGRAM_API_KEY: str = os.getenv("IDEOGRAM_API_KEY")
    PORT: int = int(os.getenv("PORT", "8000"))
    REDIS_URL: str = os.getenv("REDIS_URL")

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()


def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = RotatingFileHandler(
        f"{log_dir}/app.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


logger = setup_logging()
