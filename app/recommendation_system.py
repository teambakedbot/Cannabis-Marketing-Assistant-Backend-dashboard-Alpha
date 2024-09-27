import numpy as np
from .config import settings
from .firebase_utils import db
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone
from . import schemas
import json
from functools import lru_cache
import asyncio
from .redis_config import redis_client

# Initialize Pinecone
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index = pc.Index("product-index")

embed_model = OpenAIEmbedding(model="text-embedding-3-large")


@lru_cache(maxsize=1000)
def get_product_embedding(product_data):
    text = f"{product_data.get('name', '')} {product_data.get('category', '')} {product_data.get('description', '')}"
    embedding = embed_model.embed(text)
    return embedding


async def update_product_embeddings():
    try:
        products = db.collection("products").get()
        vectors = []
        for product in products:
            product_data = product.to_dict()
            embedding = get_product_embedding(product_data)
            vectors.append((str(product.id), embedding))

        # Use batch upsert for efficiency
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            index.upsert(vectors=batch)
    finally:
        db.close()


@lru_cache(maxsize=1000)
def get_user_preferences(user_id):
    try:
        interactions = (
            db.collection("interactions").where("user_id", "==", user_id).get()
        )
        preferences = np.zeros(100)  # Assume 100-dimensional embeddings
        for interaction in interactions:
            product = db.collection("products").document(interaction.product_id).get()
            embedding = get_product_embedding(product)
            preferences += embedding * interaction.rating

        return preferences / len(interactions) if interactions else preferences
    finally:
        db.close()


async def get_recommendations(user_id, n=5):
    cache_key = f"recommendations:{user_id}"
    cached_recommendations = await redis_client.get(cache_key)
    if cached_recommendations:
        return json.loads(cached_recommendations)

    user_preferences = get_user_preferences(user_id)
    response = index.query(vector=user_preferences, top_k=n, include_metadata=True)
    recommended_products = []
    for match in response["matches"]:
        product_id = match["id"]
        product_doc = db.collection("products").document(product_id).get()
        if product_doc.exists:
            recommended_products.append(product_doc.to_dict())

    # Cache the recommendations for 1 hour
    await redis_client.set(cache_key, json.dumps(recommended_products), ex=3600)
    return recommended_products


async def get_search_products(query: str, page: int = 1, per_page: int = 5):
    cache_key = f"search:{query}:{page}:{per_page}"
    cached_results = await redis_client.get(cache_key)
    if cached_results:
        return json.loads(cached_results)

    query = query.lower()
    query_embedding = embed_model.get_text_embedding(query)

    fetch_count = page * per_page

    response = index.query(
        vector=query_embedding, top_k=fetch_count, include_metadata=True
    )
    search_products = []

    start_index = (page - 1) * per_page
    end_index = start_index + per_page

    for match in response["matches"][start_index:end_index]:
        product_id = match["id"]
        product_doc = db.collection("products").document(product_id).get()
        if product_doc.exists:
            product_data = product_doc.to_dict()
            product_data["id"] = product_id
            product_data["updated_at"] = product_data.get("last_updated")
            search_products.append(schemas.Product(**product_data))

    total_results = len(response["matches"])
    total_pages = (total_results + per_page - 1) // per_page

    pagination = {
        "total": total_results,
        "count": len(search_products),
        "per_page": per_page,
        "current_page": page,
        "total_pages": total_pages,
    }

    result = {"products": search_products, "pagination": pagination}

    # Cache the search results for 15 minutes
    await redis_client.set(cache_key, json.dumps(result), ex=900)
    return result
