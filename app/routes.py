from .auth import (
    create_access_token,
)
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Header,
    Query,
    Request,
    Path,
    status,
)
from sqlalchemy.orm import Session
from typing import List, Optional
from . import crud, models, schemas
from .database import get_db
from .exceptions import CustomException
from .chat_service import (
    process_chat_message,
    rename_chat,
    get_chat_messages,
    archive_chat,
    delete_chat,
)
from .user_service import get_user_chats
from .auth_service import (
    logout,
    get_current_active_user,
    get_current_user_optional,
    get_firebase_user,
)
import os


router = APIRouter()


@router.post("/chat", response_model=schemas.ChatResponse)
async def process_chat(
    request: Request,
    chat_request: schemas.ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[models.User] = Depends(get_firebase_user),
):
    # Access session data
    session = request.session
    chat_id = chat_request.chat_id or session.get("chat_id")

    # Get user_id if authenticated
    user_id = current_user.id if current_user else None
    client_ip = request.client.host
    voice_type = chat_request.voice_type
    session_id = session.get("session_id") or os.urandom(16).hex()
    message = chat_request.message
    user_agent = request.headers.get("User-Agent")
    # Process chat logic
    response = await process_chat_message(
        user_id,
        chat_id,
        session_id,
        client_ip,
        message,
        user_agent,
        voice_type,
        background_tasks,
    )

    # Update session data
    session["chat_id"] = response.chat_id

    return response


@router.get("/user/chats")
async def get_user_chats_endpoint(
    current_user: models.User = Depends(get_firebase_user),
):
    return await get_user_chats(current_user.id)


@router.get("/chat/messages")
async def get_chat_messages_endpoint(
    chat_id: str = Query(..., description="The chat ID to fetch messages for"),
    current_user: models.User = Depends(get_firebase_user),
):
    return await get_chat_messages(chat_id)


@router.put("/chat/rename")
async def rename_chat_endpoint(
    chat_id: str = Query(...),
    new_name: str = Query(...),
    authorization: str = Header(None),
):
    return await rename_chat(chat_id, new_name, authorization)


@router.put("/chat/{chat_id}/archive")
async def archive_chat_endpoint(
    chat_id: str = Path(...),
    authorization: str = Header(None),
):
    return await archive_chat(chat_id, authorization)


@router.delete("/chat/{chat_id}")
async def delete_chat_endpoint(
    chat_id: str = Path(...),
    authorization: str = Header(None),
):
    return await delete_chat(chat_id, authorization)


@router.delete("/logout")
async def logout_endpoint(
    fastapi_request: Request,
    background_tasks: BackgroundTasks,
    authorization: str = Header(None),
):
    return await logout(fastapi_request, background_tasks, authorization)


@router.post("/login", response_model=schemas.Token)
def login(user_credentials: schemas.UserLogin, db: Session = Depends(get_db)):
    user = crud.authenticate_user(db, user_credentials.email, user_credentials.password)
    if not user:
        raise CustomException(status_code=400, detail="Incorrect email or password")
    access_token = create_access_token(data={"sub": user.id})
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/register", response_model=schemas.User)
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise CustomException(status_code=400, detail="Email already registered")
    return crud.create_user(db=db, user=user)


@router.put("/users/me", response_model=schemas.User)
def update_user_me(
    user: schemas.UserUpdate,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    return crud.update_user(db, current_user.id, user)


@router.post("/products/", response_model=schemas.Product)
def create_product(product: schemas.ProductCreate, db: Session = Depends(get_db)):
    return crud.create_product(db=db, product=product)


@router.get("/products/", response_model=List[schemas.Product])
def read_products(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    products = crud.get_products(db, skip=skip, limit=limit)
    return products


@router.get("/products/{product_id}", response_model=schemas.Product)
def read_product(product_id: int, db: Session = Depends(get_db)):
    db_product = crud.get_product(db, product_id=product_id)
    if db_product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return db_product


@router.put("/products/{product_id}", response_model=schemas.Product)
def update_product(
    product_id: int, product: schemas.ProductUpdate, db: Session = Depends(get_db)
):
    return crud.update_product(db, product_id, product)


@router.delete("/products/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_product(product_id: int, db: Session = Depends(get_db)):
    crud.delete_product(db, product_id)
    return {"ok": True}


@router.post("/interactions/", response_model=schemas.Interaction)
def create_interaction(
    interaction: schemas.InteractionCreate,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    return crud.create_interaction(
        db=db, interaction=interaction, user_id=current_user.id
    )


@router.get("/interactions/", response_model=List[schemas.Interaction])
def read_interactions(
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    interactions = crud.get_user_interactions(
        db, user_id=current_user.id, skip=skip, limit=limit
    )
    return interactions


@router.post("/chat/start", response_model=schemas.ChatSession)
def start_chat_session(
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    return crud.create_chat_session(db, user_id=current_user.id)


@router.post("/dispensaries/", response_model=schemas.Dispensary)
def create_dispensary(
    dispensary: schemas.DispensaryCreate, db: Session = Depends(get_db)
):
    return crud.create_dispensary(db=db, dispensary=dispensary)


@router.get("/dispensaries/", response_model=List[schemas.Dispensary])
def read_dispensaries(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    dispensaries = crud.get_dispensaries(db, skip=skip, limit=limit)
    return dispensaries


@router.get("/dispensaries/{dispensary_id}", response_model=schemas.Dispensary)
def read_dispensary(dispensary_id: int, db: Session = Depends(get_db)):
    db_dispensary = crud.get_dispensary(db, dispensary_id=dispensary_id)
    if db_dispensary is None:
        raise HTTPException(status_code=404, detail="Dispensary not found")
    return db_dispensary


@router.post("/inventory/", response_model=schemas.Inventory)
def create_inventory(inventory: schemas.InventoryCreate, db: Session = Depends(get_db)):
    return crud.create_inventory(db=db, inventory=inventory)


@router.get("/inventory/{dispensary_id}", response_model=List[schemas.Inventory])
def read_dispensary_inventory(dispensary_id: int, db: Session = Depends(get_db)):
    inventory = crud.get_dispensary_inventory(db, dispensary_id=dispensary_id)
    return inventory


# Recommendation route
@router.get("/recommendations/", response_model=List[schemas.Product])
def get_recommendations(
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    # This is a placeholder. The actual implementation would involve your recommendation algorithm.
    return crud.get_recommended_products(db, user_id=current_user.id)


# Search route
@router.get("/search/", response_model=List[schemas.Product])
def search_products(query: str, db: Session = Depends(get_db)):
    return crud.search_products(db, query=query)
