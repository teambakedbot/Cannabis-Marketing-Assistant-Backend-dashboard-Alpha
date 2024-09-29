from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# User schemas
class UserBase(BaseModel):
    email: EmailStr
    is_active: bool = True
    is_superuser: bool = False


class UserCreate(UserBase):
    password: str


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    preferences: Optional[str] = None


class User(UserBase):
    id: str
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


# Product schemas


class ProductVariation(BaseModel):
    # Define fields for variations if needed
    pass


class Product(BaseModel):
    product_name: str
    brand: Optional[str] = None
    category: Optional[str] = None
    image_url: Optional[str] = None
    menu_provider: Optional[str] = None
    medical: Optional[bool] = None
    recreational: Optional[bool] = None
    sku: Optional[str] = None
    cann_sku_id: Optional[str] = None
    raw_product_name: Optional[str] = None
    product_name: Optional[str] = None
    lowest_price: Optional[float] = None
    variations: Optional[List[ProductVariation]] = None
    id: Optional[str] = None

    # You can use Config to allow extra fields
    class Config:
        extra = "allow"


class ProductCreate(Product):
    pass


class ProductUpdate(Product):
    pass


# Effect schemas
class Effect(BaseModel):
    id: str
    name: str
    description: str


# Interaction schemas
class Interaction(BaseModel):
    id: str
    user_id: str
    product_id: str
    interaction_type: str
    timestamp: datetime
    rating: Optional[int] = Field(None, ge=1, le=5)
    review_text: Optional[str] = None


# Chat schemas
# ChatRequest is the request body for the chat endpoint may not be needed
class ChatRequest(BaseModel):
    message: str
    voice_type: str = "normal"
    chat_id: Optional[str] = None  # Optional chat ID for authenticated users


class Pagination(BaseModel):
    total: int
    count: int
    per_page: int
    current_page: int
    total_pages: int


class ProductResults(BaseModel):
    products: List[Product]
    pagination: Pagination


class RecommendedProduct(BaseModel):
    name: str
    brand: Optional[str] = None
    category: Optional[str] = None
    image_url: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    thc_percentage: Optional[float] = None
    cbd_percentage: Optional[float] = None
    strain_type: Optional[str] = None
    effects: Optional[List[str]] = None
    flavors: Optional[List[str]] = None

    class Config:
        extra = "allow"


class ChatResponse(BaseModel):
    response: str
    data: Optional[Dict[str, Any]] = None
    products: Optional[List[RecommendedProduct]] = None
    pagination: Optional[Pagination] = None
    chat_id: str
    status_messages: Optional[str] = None


# end optional


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


# Dispensary schemas
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


# Inventory schemas
class Inventory(BaseModel):
    id: str
    retailer_id: str
    product_id: str
    quantity: int = Field(..., ge=0)
    created_at: datetime
    updated_at: datetime


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


class Token(BaseModel):
    access_token: str
    token_type: str


class InteractionCreate(BaseModel):
    user_id: str
    product_id: str
    interaction_type: str
    rating: Optional[int] = Field(None, ge=1, le=5)
    review_text: Optional[str] = None


class DispensaryCreate(BaseModel):
    name: str
    address: str
    latitude: float
    longitude: float
    phone_number: str
    operating_hours: str  # JSON string of operating hours


class InventoryCreate(BaseModel):
    retailer_id: str
    product_id: str
    quantity: int = Field(..., ge=0)
    updated_at: datetime


class ChatMessageCreate(BaseModel):
    session_id: str
    content: str
    is_from_user: bool


class FeedbackCreate(BaseModel):
    message_id: str
    feedback_type: str


class MessageRetry(BaseModel):
    message_id: str


class ContactInfo(BaseModel):
    email: EmailStr
    phone: Optional[str] = None


class OrderRequest(BaseModel):
    name: str
    contact_info: ContactInfo
    cart: Dict[str, Any]


class Order(BaseModel):
    id: str
    name: str
    contact_info: ContactInfo
    cart: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    status: str


class RecommendedProduct(BaseModel):
    name: str
    brand: Optional[str] = None
    category: Optional[str] = None
    image_url: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    tch: Optional[str] = None
    cbd: Optional[str] = None
    strain_type: Optional[str] = None
    effects: Optional[List[str]] = None
    flavors: Optional[List[str]] = None
    variations: Optional[List[Any]] = None

    # Add any other fields that are consistently present in your data
    # You can use Config to allow extra fields
    class Config:
        extra = "allow"
