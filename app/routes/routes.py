from dotenv import load_dotenv
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

# Load environment variables
load_dotenv()


app = FastAPI()

app.include_router(chat.router)
app.include_router(user.router)
app.include_router(product.router)
app.include_router(dispensary.router)
app.include_router(inventory.router)
app.include_router(order.router)
app.include_router(scraper.router)

router = app.router
