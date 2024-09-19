from celery import Celery
from .config import settings
from firebase_utils import db

celery = Celery(__name__)
celery.conf.broker_url = settings.CELERY_BROKER_URL
celery.conf.result_backend = settings.CELERY_RESULT_BACKEND


@celery.task
def update_product_recommendations():
    try:
        # Logic to update product recommendations
        pass
    finally:
        db.close()


@celery.task
def sync_inventory_data():
    try:
        # Logic to sync inventory data
        pass
    finally:
        db.close()
