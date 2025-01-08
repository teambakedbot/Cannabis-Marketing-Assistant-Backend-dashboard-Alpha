from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    FunctionMessage,
)
from ..utils.firebase_utils import db
from ..config.config import logger


class FirestoreChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores messages in Firestore."""

    def __init__(self, chat_id: str):
        self.chat_id = chat_id
        self.collection = (
            db.collection("chats").document(chat_id).collection("messages")
        )

    async def add_message(self, message: BaseMessage) -> None:
        """Add a message to the chat history."""
        try:
            message_data = {
                "content": message.content,
                "type": message.type,
                "timestamp": datetime.utcnow(),
                "additional_kwargs": message.additional_kwargs,
            }
            await self.collection.add(message_data)
        except Exception as e:
            logger.error(f"Error adding message to Firestore: {e}")
            raise

    async def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add multiple messages to the chat history."""
        try:
            batch = db.batch()
            for message in messages:
                message_data = {
                    "content": message.content,
                    "type": message.type,
                    "timestamp": datetime.utcnow(),
                    "additional_kwargs": message.additional_kwargs,
                }
                message_ref = self.collection.document()
                batch.set(message_ref, message_data)
            await batch.commit()
        except Exception as e:
            logger.error(f"Error adding messages to Firestore: {e}")
            raise

    async def clear(self) -> None:
        """Clear all messages from the chat history."""
        try:
            batch = db.batch()
            docs = await self.collection.get()
            for doc in docs:
                batch.delete(doc.reference)
            await batch.commit()
        except Exception as e:
            logger.error(f"Error clearing messages from Firestore: {e}")
            raise

    async def get_messages(self) -> List[BaseMessage]:
        """Get all messages from the chat history."""
        try:
            docs = await self.collection.order_by("timestamp").get()
            messages = []
            for doc in docs:
                data = doc.to_dict()
                message_type = data.get("type")
                content = data.get("content", "")
                additional_kwargs = data.get("additional_kwargs", {})

                if message_type == "human":
                    message = HumanMessage(
                        content=content, additional_kwargs=additional_kwargs
                    )
                elif message_type == "ai":
                    message = AIMessage(
                        content=content, additional_kwargs=additional_kwargs
                    )
                elif message_type == "system":
                    message = SystemMessage(
                        content=content, additional_kwargs=additional_kwargs
                    )
                elif message_type == "function":
                    message = FunctionMessage(
                        content=content, additional_kwargs=additional_kwargs
                    )
                else:
                    logger.warning(f"Unknown message type: {message_type}")
                    continue

                messages.append(message)
            return messages
        except Exception as e:
            logger.error(f"Error getting messages from Firestore: {e}")
            raise

    @property
    async def messages(self) -> List[BaseMessage]:
        """Get all messages from the chat history."""
        return await self.get_messages()
