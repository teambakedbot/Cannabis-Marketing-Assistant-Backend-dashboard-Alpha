from pinecone import Pinecone
from llama_index.embeddings.openai import OpenAIEmbedding
from redis import Redis
from .config import settings


def get_pinecone_index():
    """Get or create Pinecone index instance."""
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index_name = "knowledge-index"
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name, dimension=1536, metric="cosine")
    return pc.Index(index_name)


def get_product_index():
    """Get Pinecone product index instance."""
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    return pc.Index("product-index")


def get_openai_embed_model():
    """Get OpenAI embedding model instance."""
    return OpenAIEmbedding(model="text-embedding-3-large")


def get_redis_client():
    """Get Redis client instance."""
    return Redis.from_url(
        settings.REDISCLOUD_URL, encoding="utf-8", decode_responses=True
    )
