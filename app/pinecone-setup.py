import pinecone

from app.recommendation_system import get_product_embedding
from .config import settings
from .models import Product

pinecone.init(
    api_key=settings.PINECONE_API_KEY, environment=settings.PINECONE_ENVIRONMENT
)


def setup_pinecone():
    if "product-embeddings" not in pinecone.list_indexes():
        pinecone.create_index(
            "product-embeddings", dimension=100
        )  # Adjust dimension as needed


def update_pinecone_index():
    index = pinecone.Index("product-embeddings")
    try:
        products = db.query(Product).all()
        batch_size = 100
        for i in range(0, len(products), batch_size):
            batch = products[i : i + batch_size]
            embeddings = [(str(p.id), get_product_embedding(p).tolist()) for p in batch]
            index.upsert(embeddings)
    finally:
        db.close()
