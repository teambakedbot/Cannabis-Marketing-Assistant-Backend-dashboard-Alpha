from ..services.recommendation_system import get_product_embedding
from ..config.config import settings
from ..utils.firebase_utils import db
from pinecone import Pinecone
from ..config.config import logger

pc = Pinecone(api_key=settings.PINECONE_API_KEY)


def setup_pinecone():
    if "product-embeddings" not in pc.list_indexes():
        pc.create_index("product-embeddings", dimension=100)


async def update_pinecone_index():
    index = pc.Index("product-embeddings")
    try:
        products = await db.collection("products").get()
        batch_size = 100
        for i in range(0, len(products), batch_size):
            batch = products[i : i + batch_size]
            embeddings = [
                (str(p.id), await get_product_embedding(p.to_dict())) for p in batch
            ]
            await index.upsert(embeddings)
    except Exception as e:
        logger.error(f"Error updating Pinecone index: {e}")
