import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .models import Product, User, Interaction
import pinecone
from .config import settings

# Initialize Pinecone
pinecone.init(
    api_key=settings.PINECONE_API_KEY, environment=settings.PINECONE_ENVIRONMENT
)
index = pinecone.Index("product-embeddings")


def get_product_embedding(product: Product):
    # This is a simplified example. In a real-world scenario, you'd use a more sophisticated method.
    text = f"{product.name} {product.category} {product.description}"
    vectorizer = TfidfVectorizer()
    embedding = vectorizer.fit_transform([text]).toarray()[0]
    return embedding


def update_product_embeddings():
    db = SessionLocal()
    try:
        products = db.query(Product).all()
        for product in products:
            embedding = get_product_embedding(product)
            index.upsert([(str(product.id), embedding.tolist())])
    finally:
        db.close()


def get_user_preferences(user_id: int):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        interactions = (
            db.query(Interaction).filter(Interaction.user_id == user_id).all()
        )

        # Simplified preference calculation
        preferences = np.zeros(100)  # Assume 100-dimensional embeddings
        for interaction in interactions:
            product = (
                db.query(Product).filter(Product.id == interaction.product_id).first()
            )
            embedding = get_product_embedding(product)
            preferences += embedding * interaction.rating

        return preferences / len(interactions) if interactions else preferences
    finally:
        db.close()


def get_recommendations(user_id: int, n=5):
    user_preferences = get_user_preferences(user_id)
    results = index.query(user_preferences.tolist(), top_k=n)

    db = SessionLocal()
    try:
        recommended_products = []
        for result in results.matches:
            product = db.query(Product).filter(Product.id == int(result.id)).first()
            if product:
                recommended_products.append(product)
        return recommended_products
    finally:
        db.close()
