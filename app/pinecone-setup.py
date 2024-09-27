from app.recommendation_system import get_product_embedding
from .config import settings
from .firebase_utils import db
from pinecone import Pinecone

pc = Pinecone(api_key=settings.PINECONE_API_KEY)


def setup_pinecone():
    if "product-embeddings" not in pc.list_indexes():
        pc.create_index("product-embeddings", dimension=100)


def update_pinecone_index():
    index = pc.Index("product-embeddings")
    try:
        products = db.collection("products").get()
        batch_size = 100
        for i in range(0, len(products), batch_size):
            batch = products[i : i + batch_size]
            embeddings = [(str(p.id), get_product_embedding(p).tolist()) for p in batch]
            index.upsert(embeddings)
    finally:
        db.close()
