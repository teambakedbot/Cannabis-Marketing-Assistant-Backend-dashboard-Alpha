import json
import os
import firebase_admin
from firebase_admin import credentials, auth
from google.cloud.firestore_v1.async_client import AsyncClient
from google.oauth2 import service_account
from fastapi import HTTPException
from ..config.config import logger, settings


db = None


def initialize_firebase():
    global db
    """Initialize Firebase using credentials from the environment."""
    if not firebase_admin._apps:
        try:
            # Get parsed credentials from settings
            creds_dict = settings.firebase_creds

            # Initialize Firebase Admin SDK with the credential dict
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
            logger.debug("Firebase credentials loaded from environment")

            # Create service_account_credentials from the dict for AsyncClient
            service_account_credentials = (
                service_account.Credentials.from_service_account_info(creds_dict)
            )

            # Initialize AsyncClient with the same credentials and project ID
            db = AsyncClient(
                credentials=service_account_credentials,
                project=service_account_credentials.project_id,
            )
            logger.debug("Firebase initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Firebase: {str(e)}")
            raise


async def verify_firebase_token(token: str):
    """Verify Firebase authentication token."""
    try:
        decoded_token = auth.verify_id_token(token)
        logger.debug("Firebase token successfully verified: %s", decoded_token)
        return decoded_token
    except Exception as e:
        logger.error(f"Error in verify_firebase_token: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid authentication token")


initialize_firebase()
