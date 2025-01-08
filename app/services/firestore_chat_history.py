from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from ..models.schemas import ChatMessage
from ..config.config import logger
from ..utils.firebase_utils import db
from google.cloud import firestore
from fastapi import HTTPException


class FirestoreChatMessageHistory(BaseChatMessageHistory):
    """Manages chat message history in Firestore with LangChain integration."""

    def __init__(self, chat_id: str):
        """Initialize with chat ID."""
        self.chat_id = chat_id
        self.collection = (
            db.collection("chats").document(chat_id).collection("messages")
        )
        self._messages_cache: Optional[List[BaseMessage]] = None

    async def add_message(self, message: BaseMessage) -> None:
        """Add a message to the store."""
        try:
            message_data = {
                "content": message.content,
                "role": "user" if isinstance(message, HumanMessage) else "assistant",
                "timestamp": datetime.utcnow(),
                "additional_kwargs": message.additional_kwargs,
            }

            # Add to Firestore
            await self.collection.add(message_data)

            # Update cache if it exists
            if self._messages_cache is not None:
                self._messages_cache.append(message)

            logger.debug(f"Added message to chat {self.chat_id}")
        except Exception as e:
            logger.error(f"Error adding message to Firestore: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to save message")

    async def clear(self) -> None:
        """Clear message history."""
        try:
            batch = db.batch()
            docs = await self.collection.stream()
            for doc in docs:
                batch.delete(doc.reference)
            await batch.commit()

            # Clear cache
            self._messages_cache = None

            logger.debug(f"Cleared message history for chat {self.chat_id}")
        except Exception as e:
            logger.error(f"Error clearing message history: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail="Failed to clear message history"
            )

    @property
    async def messages(self) -> List[BaseMessage]:
        """Retrieve messages from Firestore with caching."""
        try:
            # Return cached messages if available
            if self._messages_cache is not None:
                return self._messages_cache

            docs = await self.collection.order_by("timestamp").stream()
            messages = []

            for doc in docs:
                data = doc.to_dict()
                additional_kwargs = data.get("additional_kwargs", {})

                if data["role"] == "user":
                    messages.append(
                        HumanMessage(
                            content=data["content"], additional_kwargs=additional_kwargs
                        )
                    )
                else:
                    messages.append(
                        AIMessage(
                            content=data["content"], additional_kwargs=additional_kwargs
                        )
                    )

            # Cache the messages
            self._messages_cache = messages
            return messages
        except Exception as e:
            logger.error(f"Error retrieving messages: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to retrieve messages")

    async def get_messages(self, limit: Optional[int] = None) -> List[ChatMessage]:
        """Get messages with optional limit."""
        try:
            query = self.collection.order_by(
                "timestamp", direction=firestore.Query.DESCENDING
            )
            if limit:
                query = query.limit(limit)

            docs = await query.get()
            messages = []

            for doc in docs:
                message_data = doc.to_dict()
                message_data["id"] = doc.id
                message_data["chat_id"] = self.chat_id
                messages.append(ChatMessage(**message_data))

            return list(reversed(messages))
        except Exception as e:
            logger.error(f"Error retrieving messages: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to retrieve messages")

    async def add_user_message(
        self, content: str, user_id: Optional[str] = None
    ) -> None:
        """Add a user message."""
        message = HumanMessage(
            content=content, additional_kwargs={"user_id": user_id} if user_id else {}
        )
        await self.add_message(message)

    async def add_ai_message(self, content: str, is_streaming: bool = False) -> None:
        """Add an AI message."""
        message = AIMessage(
            content=content, additional_kwargs={"is_streaming": is_streaming}
        )
        await self.add_message(message)

    async def update_chat_metadata(self, metadata: Dict[str, Any]) -> None:
        """Update chat document metadata."""
        try:
            chat_ref = db.collection("chats").document(self.chat_id)
            metadata["updated_at"] = datetime.utcnow()
            await chat_ref.set(metadata, merge=True)
            logger.debug(f"Updated metadata for chat {self.chat_id}")
        except Exception as e:
            logger.error(f"Error updating chat metadata: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail="Failed to update chat metadata"
            )
