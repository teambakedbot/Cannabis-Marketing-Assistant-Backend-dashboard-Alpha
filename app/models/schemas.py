from pydantic import BaseModel, EmailStr, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


# User schemas
class UserBase(BaseModel):
    email: EmailStr
    is_active: bool = True
    is_superuser: bool = False


class UserCreate(UserBase):
    password: str


class UserUpdate(BaseModel):
    email: EmailStr | None = None
    full_name: str | None = None
    date_of_birth: datetime | None = None
    preferences: str | None = None


class User(UserBase):
    id: str
    created_at: datetime
    updated_at: datetime
    last_login: datetime | None = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Product(BaseModel):
    cann_sku_id: str
    brand_name: str | None = None
    brand_id: int | None = None
    url: str | None = None
    image_url: str | None = None
    raw_product_name: str
    product_name: str
    raw_weight_string: str | None = None
    display_weight: str | None = None
    raw_product_category: str | None = None
    category: str
    raw_subcategory: str | None = None
    subcategory: str | None = None
    product_tags: List[str] | None = []
    percentage_thc: float | None = None
    percentage_cbd: float | None = None
    mg_thc: float | None = None
    mg_cbd: float | None = None
    quantity_per_package: int | None = None
    medical: bool
    recreational: bool
    latest_price: float
    menu_provider: str
    retailer_id: str
    meta_sku: str

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
    rating: int | None = Field(None, ge=1, le=5)
    review_text: str | None = None


# Chat schemas
class ChatRequest(BaseModel):
    message: str
    voice_type: str = "normal"
    chat_id: str | None = None


class Pagination(BaseModel):
    total: int
    count: int
    per_page: int
    current_page: int
    total_pages: int


class GroupedProduct(BaseModel):
    meta_sku: str
    retailer_id: str
    products: List[Product]


class ProductResults(BaseModel):
    products: List[GroupedProduct]
    pagination: Pagination


class ChatSession(BaseModel):
    id: str
    user_id: str
    start_time: datetime
    end_time: datetime | None = None
    session_data: str | None = None


class ChatMessage(BaseModel):
    chat_id: str
    message_id: str
    user_id: str | None = None
    session_id: str
    role: str
    content: str
    data: Dict[str, Any] | None = None
    timestamp: datetime | None = None


# Dispensary schemas
class Dispensary(BaseModel):
    id: str
    name: str
    address: str
    latitude: float
    longitude: float
    phone_number: str
    operating_hours: str
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
    rating: int | None = Field(None, ge=1, le=5)
    review_text: str | None = None


class DispensaryCreate(BaseModel):
    name: str
    address: str
    latitude: float
    longitude: float
    phone_number: str
    operating_hours: str


class InventoryCreate(BaseModel):
    retailer_id: str
    product_id: str
    quantity: int = Field(..., ge=0)
    updated_at: datetime


class ChatMessageCreate(BaseModel):
    session_id: str
    content: str
    role: str


class FeedbackCreate(BaseModel):
    message_id: str
    feedback_type: str


class MessageRetry(BaseModel):
    message_id: str


class ContactInfo(BaseModel):
    email: EmailStr
    phone: str | None = None


class OrderRequest(BaseModel):
    name: str
    contact_info: ContactInfo
    cart: Dict[str, Any]
    total_price: float


class Order(BaseModel):
    id: str
    name: str
    contact_info: ContactInfo
    cart: Dict[str, Any]
    created_at: datetime
    updated_at: datetime | None = None
    status: str

    class Config:
        extra = "allow"


class GemmaChatRequest(BaseModel):
    message: str
    max_length: Optional[int] = 256


class GemmaChatResponse(BaseModel):
    response: str
    model: str = "GemmaLM-for-Cannabis"
