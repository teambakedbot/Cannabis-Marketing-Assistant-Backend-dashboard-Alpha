from sqlalchemy.orm import Session
from . import models, schemas
from datetime import datetime
from .exceptions import CustomException
from typing import List
from passlib.context import CryptContext
from fastapi import HTTPException

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def authenticate_user(db: Session, email: str, password: str):
    user = get_user_by_email(db, email)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = models.User(email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


# User CRUD operations
def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()


def update_user(db: Session, user_id: int, user: schemas.UserUpdate):
    db_user = get_user(db, user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    update_data = user.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_user, key, value)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


# Product CRUD operations
def get_product(db: Session, product_id: int):
    return db.query(models.Product).filter(models.Product.id == product_id).first()


def get_products(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Product).offset(skip).limit(limit).all()


def create_product(db: Session, product: schemas.ProductCreate):
    db_product = models.Product(**product.dict())
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    return db_product


def update_product(db: Session, product_id: int, product: schemas.ProductUpdate):
    db_product = get_product(db, product_id)
    if not db_product:
        raise HTTPException(status_code=404, detail="Product not found")
    update_data = product.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_product, key, value)
    db_product.updated_at = datetime.utcnow()
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    return db_product


def delete_product(db: Session, product_id: int):
    db_product = get_product(db, product_id)
    if not db_product:
        raise HTTPException(status_code=404, detail="Product not found")
    db.delete(db_product)
    db.commit()
    return db_product


# Interaction CRUD operations
def create_interaction(
    db: Session, interaction: schemas.InteractionCreate, user_id: int
):
    db_interaction = models.Interaction(**interaction.dict(), user_id=user_id)
    db.add(db_interaction)
    db.commit()
    db.refresh(db_interaction)
    return db_interaction


def get_user_interactions(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    return (
        db.query(models.Interaction)
        .filter(models.Interaction.user_id == user_id)
        .offset(skip)
        .limit(limit)
        .all()
    )


# Chat CRUD operations
def create_chat_session(db: Session, user_id: int):
    db_chat_session = models.ChatSession(user_id=user_id)
    db.add(db_chat_session)
    db.commit()
    db.refresh(db_chat_session)
    return db_chat_session


def create_chat_message(
    db: Session, session_id: int, message: schemas.ChatMessageCreate
):
    db_message = models.ChatMessage(**message.dict(), session_id=session_id)
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message


def get_chat_messages(db: Session, session_id: int):
    return (
        db.query(models.ChatMessage)
        .filter(models.ChatMessage.session_id == session_id)
        .order_by(models.ChatMessage.timestamp)
        .all()
    )


# Dispensary CRUD operations
def create_dispensary(db: Session, dispensary: schemas.DispensaryCreate):
    db_dispensary = models.Dispensary(**dispensary.dict())
    db.add(db_dispensary)
    db.commit()
    db.refresh(db_dispensary)
    return db_dispensary


def get_dispensaries(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Dispensary).offset(skip).limit(limit).all()


def get_dispensary(db: Session, dispensary_id: int):
    return (
        db.query(models.Dispensary)
        .filter(models.Dispensary.id == dispensary_id)
        .first()
    )


# Inventory CRUD operations
def create_inventory(db: Session, inventory: schemas.InventoryCreate):
    db_inventory = models.Inventory(**inventory.dict())
    db.add(db_inventory)
    db.commit()
    db.refresh(db_inventory)
    return db_inventory


def get_dispensary_inventory(db: Session, dispensary_id: int):
    return (
        db.query(models.Inventory)
        .filter(models.Inventory.dispensary_id == dispensary_id)
        .all()
    )


# Recommendation function (placeholder)
def get_recommended_products(db: Session, user_id: int) -> List[models.Product]:
    # This is a placeholder. In a real implementation, you would use your recommendation algorithm here.
    # For now, we'll just return the 5 most recently added products
    return (
        db.query(models.Product)
        .order_by(models.Product.created_at.desc())
        .limit(5)
        .all()
    )


# Search function (placeholder)
def search_products(db: Session, query: str) -> List[models.Product]:
    # This is a placeholder. In a real implementation, you would use your search algorithm here.
    # For now, we'll just do a simple case-insensitive search on the product name
    return (
        db.query(models.Product).filter(models.Product.name.ilike(f"%{query}%")).all()
    )
