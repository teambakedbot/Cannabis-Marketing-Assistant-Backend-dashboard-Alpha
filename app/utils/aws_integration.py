import boto3
from app import settings


# Initialize AWS clients
comprehend = boto3.client(
    "comprehend",
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION,
)

personalize = boto3.client(
    "personalize",
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION,
)


def analyze_sentiment(text):
    response = comprehend.detect_sentiment(Text=text, LanguageCode="en")
    return response["Sentiment"]


def get_personalized_recommendations(user_id, campaign_arn):
    response = personalize.get_recommendations(
        campaignArn=campaign_arn, userId=str(user_id)
    )
    return response["itemList"]


# Add more AWS service integrations as needed
