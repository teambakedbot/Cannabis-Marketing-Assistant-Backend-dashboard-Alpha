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
from ..config.config import logger, settings
from langchain_openai import ChatOpenAI
import logging

# Configure logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class FirestoreChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores messages in Firestore."""

    def __init__(self, chat_id: str):
        logger.debug(f"Initializing FirestoreChatMessageHistory for chat_id: {chat_id}")
        self.chat_id = chat_id
        self.collection = (
            db.collection("chats").document(chat_id).collection("messages")
        )

    async def add_message(self, message: BaseMessage) -> None:
        """Add a message to the chat history."""
        try:
            logger.debug(
                f"Adding message of type {message.type} to chat {self.chat_id}"
            )
            message_data = {
                "content": message.content,
                "type": message.type,
                "timestamp": datetime.utcnow(),
                "additional_kwargs": message.additional_kwargs,
            }
            await self.collection.add(message_data)
            logger.debug(f"Successfully added message to chat {self.chat_id}")
        except Exception as e:
            logger.error(f"Error adding message to Firestore: {e}", exc_info=True)
            raise

    async def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add multiple messages to the chat history."""
        try:
            logger.debug(f"Adding {len(messages)} messages to chat {self.chat_id}")
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
            logger.debug(f"Successfully added {len(messages)} messages in batch")
        except Exception as e:
            logger.error(f"Error adding messages to Firestore: {e}", exc_info=True)
            raise

    async def clear(self) -> None:
        """Clear all messages from the chat history."""
        try:
            logger.debug(f"Clearing all messages from chat {self.chat_id}")
            batch = db.batch()
            docs = await self.collection.get()
            for doc in docs:
                batch.delete(doc.reference)
            await batch.commit()
            logger.debug(f"Successfully cleared {len(docs)} messages")
        except Exception as e:
            logger.error(f"Error clearing messages from Firestore: {e}", exc_info=True)
            raise

    async def get_messages(self) -> List[BaseMessage]:
        """Get all messages from the chat history."""
        try:
            logger.debug(f"Retrieving messages for chat {self.chat_id}")
            # Get messages ordered by timestamp
            docs = await self.collection.order_by("timestamp").get()
            messages = []
            logger.debug(f"Found {len(docs)} messages in Firestore")

            # Convert messages to appropriate types
            for doc in docs:
                data = doc.to_dict()
                message_type = data.get("type")
                content = data.get("content", "")
                additional_kwargs = data.get("additional_kwargs", {})

                logger.debug(f"Converting message of type: {message_type}")
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

            logger.debug(f"Successfully converted {len(messages)} messages")
            return messages
        except Exception as e:
            logger.error(f"Error getting messages from Firestore: {e}", exc_info=True)
            raise

    @property
    async def messages(self) -> List[BaseMessage]:
        """Get all messages from the chat history."""
        return await self.get_messages()


class SummarizingFirestoreChatMessageHistory(FirestoreChatMessageHistory):
    """Chat message history that stores messages in Firestore with summarization."""

    def __init__(self, chat_id: str, max_recent_messages: int = 20):
        logger.debug(
            f"Initializing SummarizingFirestoreChatMessageHistory for chat_id: {chat_id}"
        )
        super().__init__(chat_id)
        self.max_recent_messages = max_recent_messages
        self.llm = ChatOpenAI(
            model_name=settings.OPENAI_MODEL_NAME,
            temperature=0.1,
            max_tokens=4096,
        )
        logger.debug(f"Initialized with max_recent_messages: {max_recent_messages}")

    async def get_messages(self) -> List[BaseMessage]:
        """Get messages with summarization for older messages."""
        try:
            logger.debug(
                f"Retrieving messages with summarization for chat {self.chat_id}"
            )
            # Get all messages ordered by timestamp for this specific chat
            docs = await self.collection.order_by("timestamp").get()
            messages = []
            logger.debug(f"Found {len(docs)} total messages")

            # If this is a new chat or no messages exist, return empty list
            if not docs:
                logger.debug("No messages found, returning empty list")
                return []

            # If we have more messages than max_recent_messages, summarize older ones
            if len(docs) > self.max_recent_messages:
                logger.debug(
                    f"Messages exceed max_recent_messages ({len(docs)} > {self.max_recent_messages})"
                )
                # Get older messages for summarization
                older_messages = docs[: -self.max_recent_messages]
                recent_messages = docs[-self.max_recent_messages :]
                logger.debug(
                    f"Split into {len(older_messages)} older and {len(recent_messages)} recent messages"
                )

                # Convert older messages to text for summarization
                older_content = []
                for doc in older_messages:
                    data = doc.to_dict()
                    role = data.get("type", "")
                    content = data.get("content", "")
                    # Skip system messages in the summary
                    if role != "system":
                        older_content.append(f"{role}: {content}")

                if older_content:  # Only summarize if there are messages to summarize
                    logger.debug(f"Summarizing {len(older_content)} older messages")
                    # Join messages with newlines for the prompt
                    message_history = "\n".join(older_content)

                    # Generate summary using LLM
                    summary_prompt = [
                        SystemMessage(
                            content="You are a helpful assistant that creates concise summaries of conversations. Focus on key points and decisions made."
                        ),
                        HumanMessage(
                            content=f"Please summarize this conversation history. Focus on the key points and any important decisions or information shared:\n\n{message_history}"
                        ),
                    ]

                    logger.debug("Generating conversation summary using LLM")
                    summary_response = await self.llm.ainvoke(summary_prompt)
                    logger.debug("Successfully generated summary")

                    # Add summary as a system message
                    messages.append(
                        SystemMessage(
                            content=f"Summary of previous conversation:\n{summary_response.content}"
                        )
                    )
                    logger.debug("Added summary as system message")

                # Process recent messages
                messages_to_process = recent_messages
                logger.debug("Processing recent messages")
            else:
                messages_to_process = docs
                logger.debug("Processing all messages (no summarization needed)")

            # Convert messages to appropriate types
            converted_count = 0
            for doc in messages_to_process:
                data = doc.to_dict()
                message_type = data.get("type")
                content = data.get("content", "")
                additional_kwargs = data.get("additional_kwargs", {})

                # Ensure the message belongs to this chat
                if additional_kwargs.get("chat_id") != self.chat_id:
                    logger.warning(
                        f"Skipping message with mismatched chat_id: {additional_kwargs.get('chat_id')}"
                    )
                    continue

                logger.debug(f"Converting message of type: {message_type}")
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
                converted_count += 1

            logger.debug(f"Successfully converted {converted_count} messages")
            logger.debug(f"Final message list contains {len(messages)} messages")
            return messages

        except Exception as e:
            logger.error(f"Error getting messages from Firestore: {e}", exc_info=True)
            raise
