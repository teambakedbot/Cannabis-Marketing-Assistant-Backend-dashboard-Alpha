from pydantic import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://user:password@localhost/dbname"
    REDIS_URL: str = "redis://localhost"
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALLOWED_ORIGINS: list = ["*"]
    DEBUG: bool = True
    WORKERS: int = 1
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    PINECONE_API_KEY: str = "your-pinecone-api-key"
    PINECONE_ENVIRONMENT: str = "your-pinecone-environment"
    AWS_ACCESS_KEY_ID: str = "your-aws-access-key-id"
    AWS_SECRET_ACCESS_KEY: str = "your-aws-secret-access-key"
    AWS_REGION: str = "your-aws-region"

    class Config:
        env_file = ".env"

settings = Settings()
