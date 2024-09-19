from requests import Session
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from . import settings
from fastapi import Depends
from .models import User

SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(user_id: str, db: Session = Depends(get_db)):
    # Logic to retrieve the current user from the database
    # Example: return db.query(User).filter(User.id == user_id).first()
    return db.query(User).filter(User.id == user_id).first()
