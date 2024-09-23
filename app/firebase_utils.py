import json
import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, auth, firestore
import logging
from fastapi import HTTPException
from .config import logger

load_dotenv()


def initialize_firebase():
    """Initialize Firebase using credentials from the environment."""
    if not firebase_admin._apps:
        cred_input = os.getenv("FIREBASE_CREDENTIALS")
        if not cred_input:
            raise ValueError("FIREBASE_CREDENTIALS environment variable is not set")

        try:
            cred_json = json.loads(cred_input)
            cred = credentials.Certificate(cred_json)
            logger.debug("Firebase credentials loaded from JSON")
        except json.JSONDecodeError:
            if not os.path.exists(cred_input):
                raise FileNotFoundError(f"Credential path {cred_input} does not exist")
            cred = credentials.Certificate(cred_input)
            logger.debug("Firebase credentials loaded from file: %s", cred_input)

        firebase_admin.initialize_app(cred)
        logger.debug("Firebase initialized successfully")


def verify_firebase_token(token: str):
    """Verify Firebase authentication token."""
    try:
        decoded_token = auth.verify_id_token(token)
        logger.debug("Firebase token successfully verified: %s", decoded_token)
        return decoded_token
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication token")


initialize_firebase()
db = firestore.client()
