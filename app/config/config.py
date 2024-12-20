import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv(override=True)


class Settings:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
    CANNMENUS_API_KEY = os.getenv("CANNMENUS_API_KEY")
    IDEOGRAM_API_KEY = os.getenv("IDEOGRAM_API_KEY")

    # Database settings
    FIREBASE_CREDENTIALS = os.getenv("FIREBASE_CREDENTIALS")

    # OpenAI Model settings
    OPENAI_MODEL_NAME = os.getenv(
        "OPENAI_MODEL_NAME", "ft:gpt-4o-mini-2024-07-18:bakedbot::AfmUOhnt"
    )

    # Redis settings
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    REDISCLOUD_URL: str = os.getenv("REDISCLOUD_URL")

    # Email settings
    SENDGRID_FROM_EMAIL = os.getenv("SENDGRID_FROM_EMAIL")
    RETAILER_EMAIL = os.getenv("RETAILER_EMAIL")

    # SMS settings
    TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

    # Application settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Rate limiting
    RATE_LIMIT = os.getenv("RATE_LIMIT", "100/minute")
    SECRET_KEY = os.getenv("SECRET_KEY")
    ENVIRONMENT = os.getenv("ENVIRONMENT")
    FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET")


settings = Settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Log the configuration (be careful not to log sensitive information)
logger.info(f"Debug mode: {settings.DEBUG}")
logger.info(f"Log level: {settings.LOG_LEVEL}")
logger.info(f"Rate limit: {settings.RATE_LIMIT}")
