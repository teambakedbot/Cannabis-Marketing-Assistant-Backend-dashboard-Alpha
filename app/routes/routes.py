from app.routes import (
    chat,
    user,
    product,
    dispensary,
    inventory,
    order,
    scraper,
)
from fastapi import FastAPI

app = FastAPI(
    title="Smokey API",
    description="API for Smokey, an AI-powered cannabis product recommendation system",
    version="1.0.0",
)

app.include_router(chat.router)
app.include_router(user.router)
app.include_router(product.router)
app.include_router(dispensary.router)
app.include_router(inventory.router)
app.include_router(order.router)
app.include_router(scraper.router)

router = app.router
