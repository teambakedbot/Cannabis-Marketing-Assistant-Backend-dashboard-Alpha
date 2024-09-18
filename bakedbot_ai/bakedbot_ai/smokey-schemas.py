from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
from datetime import datetime

# User schemas
class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    preferences: Optional[str] = None

class User(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]

    class Config:
        orm_mode = True

# Product schemas
class ProductBase(BaseModel):
    name: str
    category: str
    thc_content: float = Field(..., ge=0, le=100)
    cbd_content: float = Field(..., ge=0, le=100)
    description: str
    price: float = Field(..., ge=0)
    stock_quantity: int = Field(..., ge=0)

class ProductCreate(ProductBase):
    pass

class ProductUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    thc_content: Optional[float] = Field(None, ge=0, le=100)
    cbd_content: Optional[float] = Field(None, ge=0, le=100)
    description: Optional[str] = None
    price: Optional[float] = Field(None, ge=0)
    stock_quantity: Optional[int] = Field(None, ge=0)

class Product(ProductBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

# Effect schemas
class EffectBase(BaseModel):
    name: str
    description: str

class EffectCreate(EffectBase):
    pass

class Effect(EffectBase):
    id: int

    class Config:
        orm_mode = True

# Interaction schemas
class InteractionBase(BaseModel):
    product_id: int
    interaction_type: str
    rating: Optional[int] = Field(None, ge=1, le=5)
    review_text: Optional[str] = None

class InteractionCreate(InteractionBase):
    pass

class Interaction(InteractionBase):
    id: int
    user_id: int
    timestamp: datetime

    class Config:
        orm_mode = True

# Chat schemas
class ChatSessionBase(BaseModel):
    user_id: int

class ChatSessionCreate(ChatSessionBase):
    pass

class ChatSession(ChatSessionBase):
    id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    session_data: Optional[str] = None

    class Config:
        orm_mode = True

class ChatMessageBase(BaseModel):
    content: str
    is_from_user: bool

class ChatMessageCreate(ChatMessageBase):
    pass

class ChatMessage(ChatMessageBase):
    id: int
    session_id: int
    timestamp: datetime

    class Config:
        orm_mode = True

# Dispensary schemas
class DispensaryBase(BaseModel):
    name: str
    address: str
    latitude: float
    longitude: float
    phone_number: str
    operating_hours: str  # JSON string of operating hours

class DispensaryCreate(DispensaryBase):
    pass

class Dispensary(DispensaryBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

# Inventory schemas
class InventoryBase(BaseModel):
    dispensary_id: int
    product_id: int
    quantity: int = Field(..., ge=0)

class InventoryCreate(InventoryBase):
    pass

class Inventory(InventoryBase):
    id: int
    last_updated: datetime

    class Config:
        orm_mode = True

# Recommendation schemas
class RecommendationRequest(BaseModel):
    user_id: int

class RecommendationResponse(BaseModel):
    products: List[Product]

# Search schemas
class SearchRequest(BaseModel):
    query: str

class SearchResponse(BaseModel):
    products: List[Product]

