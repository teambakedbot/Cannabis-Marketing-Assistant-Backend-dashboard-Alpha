from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    voice_type: str = "normal"
    chat_id: str = None  # Optional chat ID for authenticated users

class ChatResponse(BaseModel):
    response: str
    chat_id: str | None = None  # Include chat_id in the response
