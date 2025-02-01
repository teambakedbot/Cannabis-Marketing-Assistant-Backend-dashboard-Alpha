import os
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_community.cache import RedisCache, BaseCache
import langchain
from redis import Redis
from ..utils.logging_config import setup_logging, get_logger
import json
import ast

from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings."""

    # API Keys and Credentials
    OPENAI_API_KEY: str
    PINECONE_API_KEY: str
    IDEOGRAM_API_KEY: str
    FIREBASE_STORAGE_BUCKET: str
    FIREBASE_CREDENTIALS: object
    CANNMENUS_API_KEY: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    SENDGRID_API_KEY: str
    TWILIO_ACCOUNT_SID: str
    TWILIO_AUTH_TOKEN: str
    TWILIO_DEFAULT_FROM_NUMBER: str

    # Service URLs and Configuration
    REDISCLOUD_URL: str
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    PINECONE_ENVIRONMENT: str = "us-east-1"

    # Email Configuration
    SENDGRID_FROM_EMAIL: str
    RETAILER_EMAIL: str

    # Security
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # LLM Configuration
    OPENAI_MODEL_NAME: str = "gpt-4-turbo-preview"
    OPENAI_TEMPERATURE: float = 0.1
    OPENAI_MAX_TOKENS: int = 4096

    # Application Configuration
    PORT: int = 8080
    DEBUG: bool = True
    WORKERS: int = 1
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    ENVIRONMENT: str = "development"

    # Cache Configuration
    CACHE_TTL: int = 3600  # 1 hour
    MAX_CACHE_SIZE: int = 100

    # Rate Limiting
    RATE_LIMIT_CALLS: int = 100
    RATE_LIMIT_PERIOD: int = 60  # seconds

    @property
    def firebase_creds(self) -> dict:
        """Parse and return Firebase credentials as dict."""
        try:
            creds_input = self.FIREBASE_CREDENTIALS

            # If it's already a dict, return it
            if isinstance(creds_input, dict):
                return creds_input

            # If it's a string, try to parse it as JSON first
            if isinstance(creds_input, str):
                # Clean the string: remove surrounding quotes and handle escaped newlines
                cleaned_input = creds_input.strip()
                if cleaned_input.startswith("'") and cleaned_input.endswith("'"):
                    cleaned_input = cleaned_input[1:-1]
                elif cleaned_input.startswith('"') and cleaned_input.endswith('"'):
                    cleaned_input = cleaned_input[1:-1]

                # Try parsing as JSON first
                try:
                    return json.loads(cleaned_input)
                except json.JSONDecodeError:
                    # If JSON parsing fails and it looks like a file path, try reading the file
                    if cleaned_input.endswith(".json") or "/" in cleaned_input:
                        if not os.path.exists(cleaned_input):
                            raise ValueError(
                                f"Firebase credentials file not found: {cleaned_input}"
                            )
                        with open(cleaned_input, "r") as f:
                            return json.load(f)
                    else:
                        # If it's not a valid JSON string and not a file path, raise error
                        raise ValueError(
                            "Invalid JSON string format for Firebase credentials"
                        )

            raise ValueError(
                "Firebase credentials must be a JSON string, file path, or dict"
            )
        except Exception as e:
            logger.error(f"Error parsing Firebase credentials: {e}")
            raise ValueError("Invalid Firebase credentials format") from e

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() == "production"

    def get_cache(self) -> BaseCache:
        """Get Redis cache instance."""
        redis_client = Redis.from_url(
            self.REDISCLOUD_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
            retry_on_timeout=True,
        )
        return RedisCache(redis_client=redis_client)

    @property
    def llm(self) -> ChatOpenAI:
        """Get configured LLM instance with caching."""
        # Initialize Redis cache for LLM if not already set
        if not hasattr(langchain, "llm_cache") or langchain.llm_cache is None:
            langchain.llm_cache = self.get_cache()

        return ChatOpenAI(
            model_name=self.OPENAI_MODEL_NAME,
            temperature=self.OPENAI_TEMPERATURE,
            max_tokens=self.OPENAI_MAX_TOKENS,
            cache=True,  # Enable caching
            streaming=True,  # Enable streaming for better performance
            request_timeout=60.0,  # Set timeout
            max_retries=3,  # Add retries
        )

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings instance
    """
    return Settings()


# Initialize settings
settings = get_settings()

# Configure logging
setup_logging(
    log_level=settings.LOG_LEVEL,
    log_file=settings.LOG_FILE,
    json_format=settings.is_production,
)

# Get logger instance
logger = get_logger(__name__)
