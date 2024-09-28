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
    cann_sku_id: str
    brand_name: Optional[str] = Field(None)
    brand_id: Optional[int] = Field(None)
    url: Optional[str] = Field(None)
    image_url: Optional[str] = Field(None)
    raw_product_name: str
    product_name: str
    raw_weight_string: Optional[str] = Field(None)
    display_weight: Optional[str] = Field(None)
    raw_product_category: Optional[str] = Field(None)
    category: str
    raw_subcategory: Optional[str] = Field(None)
    subcategory: Optional[str] = Field(None)
    product_tags: Optional[List[str]] = Field(None)
    percentage_thc: Optional[float] = Field(None)
    percentage_cbd: Optional[float] = Field(None)
    mg_thc: Optional[float] = Field(None)
    mg_cbd: Optional[float] = Field(None)
    quantity_per_package: Optional[int] = Field(None)
    medical: bool
    recreational: bool
    latest_price: float
    menu_provider: str


class ProductBase(BaseModel):
    retailer_id: int
    sku: str
    variations: List[ProductVariation]
    updated_at: datetime


class ProductCreate(ProductBase):
    pass


class ProductUpdate(ProductBase):
    pass


class Product(ProductBase):
    id: str
    created_at: datetime
    updated_at: datetime


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


class ChatResponse(BaseModel):
    response: str
    data: Optional[Dict[str, List[Product]]] = None
    pagination: Optional[Pagination] = None
    chat_id: str


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
