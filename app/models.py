from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime


class UserBase(BaseModel):
    email: EmailStr
    is_active: bool = True
    is_superuser: bool = False


class User(UserBase):
    id: str
    created_at: datetime
    last_login: Optional[datetime] = None


class UserProfile(BaseModel):
    user_id: str
    full_name: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    preferences: Optional[str] = None


class ProductBase(BaseModel):
    name: str
    category: str
    thc_content: float = Field(..., ge=0, le=100)
    cbd_content: float = Field(..., ge=0, le=100)
    description: str
    price: float = Field(..., ge=0)
    stock_quantity: int = Field(..., ge=0)


class Product(ProductBase):
    id: str
    created_at: datetime
    updated_at: datetime


class Effect(BaseModel):
    id: str
    name: str
    description: str


class Interaction(BaseModel):
    id: str
    user_id: str
    product_id: str
    interaction_type: str
    timestamp: datetime
    rating: Optional[int] = Field(None, ge=1, le=5)
    review_text: Optional[str] = None


class ChatSession(BaseModel):
    id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    session_data: Optional[str] = None


class ChatMessage(BaseModel):
    id: str
    session_id: str
    content: str
    timestamp: datetime
    is_from_user: bool


class Dispensary(BaseModel):
    id: str
    name: str
    address: str
    latitude: float
    longitude: float
    phone_number: str
    operating_hours: str  # JSON string of operating hours
    created_at: datetime
    updated_at: datetime


class Inventory(BaseModel):
    id: str
    dispensary_id: str
    product_id: str
    quantity: int = Field(..., ge=0)
    last_updated: datetime
