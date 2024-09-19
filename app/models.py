from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Table
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

# Association table for many-to-many relationship between products and effects
product_effect = Table('product_effect', Base.metadata,
    Column('product_id', Integer, ForeignKey('products.id')),
    Column('effect_id', Integer, ForeignKey('effects.id'))
)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    profile = relationship("UserProfile", back_populates="user", uselist=False)
    interactions = relationship("Interaction", back_populates="user")

class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    full_name = Column(String)
    date_of_birth = Column(DateTime)
    preferences = Column(String)  # JSON string of preferences

    user = relationship("User", back_populates="profile")

class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    category = Column(String, index=True)
    thc_content = Column(Float)
    cbd_content = Column(Float)
    description = Column(String)
    price = Column(Float)
    stock_quantity = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    effects = relationship("Effect", secondary=product_effect, back_populates="products")
    interactions = relationship("Interaction", back_populates="product")

class Effect(Base):
    __tablename__ = "effects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String)

    products = relationship("Product", secondary=product_effect, back_populates="effects")

class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    product_id = Column(Integer, ForeignKey("products.id"))
    interaction_type = Column(String)  # e.g., 'view', 'purchase', 'review'
    timestamp = Column(DateTime, default=datetime.utcnow)
    rating = Column(Integer, nullable=True)
    review_text = Column(String, nullable=True)

    user = relationship("User", back_populates="interactions")
    product = relationship("Product", back_populates="interactions")

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    session_data = Column(String)  # JSON string of session data

    messages = relationship("ChatMessage", back_populates="session")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    content = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    is_from_user = Column(Boolean)

    session = relationship("ChatSession", back_populates="messages")

class Dispensary(Base):
    __tablename__ = "dispensaries"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    address = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    phone_number = Column(String)
    operating_hours = Column(String)  # JSON string of operating hours
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    inventory = relationship("Inventory", back_populates="dispensary")

class Inventory(Base):
    __tablename__ = "inventory"

    id = Column(Integer, primary_key=True, index=True)
    dispensary_id = Column(Integer, ForeignKey("dispensaries.id"))
    product_id = Column(Integer, ForeignKey("products.id"))
    quantity = Column(Integer)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    dispensary = relationship("Dispensary", back_populates="inventory")
    product = relationship("Product")

