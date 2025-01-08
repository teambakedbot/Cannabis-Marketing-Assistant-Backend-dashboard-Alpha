from datetime import datetime
from typing import Optional, AsyncGenerator, Dict, Any, TypedDict, List
from fastapi import HTTPException
import asyncio

# Core components
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import StreamingStdOutCallbackHandler

# LangGraph components
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Main langchain package
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferMemory

from ..config.config import logger, settings
from ..models.schemas import ChatMessage
from ..utils.firebase_utils import db
from ..tools.tools import configurable_agent, llm
from ..utils.sessions import clean_up_old_sessions
from ..utils.redis_config import FirestoreEncoder
from ..services.firestore_chat_history import FirestoreChatMessageHistory

from firebase_admin import firestore
from redis.asyncio import Redis
import json
import os
from functools import lru_cache
from fastapi import BackgroundTasks


class MessagesState(TypedDict):
    """State definition for the chat graph."""

    messages: List[Dict[str, Any]]  # Stores the actual conversation messages
    metadata: Dict[str, Any]  # Stores chat metadata like user_id, session_id, etc.
    next_step: str  # Controls workflow routing
    agent_scratchpad: List[Dict[str, Any]]  # For tool outputs and intermediate steps
    chat_id: str  # Identifies the conversation thread


async def process_chat_message(
    user_id: Optional[str],
    chat_id: str,
    session_id: str,
    client_ip: str,
    message: str,
    user_agent: str,
    voice_type: str,
    background_tasks: BackgroundTasks,
    redis_client: Redis,
    language: str = "English",
) -> AsyncGenerator[ChatMessage, None]:
    """Process a chat message and stream the AI response."""
    try:
        # Create message queue for streaming
        message_queue = asyncio.Queue()

        # Create streaming callback handler
        class StreamingHandler(StreamingStdOutCallbackHandler):
            def __init__(self, chat_id: str, queue: asyncio.Queue):
                super().__init__()
                self.tokens = []
                self.chat_id = chat_id
                self.queue = queue

            def on_llm_new_token(self, token: str, **kwargs) -> None:
                self.tokens.append(token)
                # Create streaming message
                streaming_message = ChatMessage(
                    chat_id=self.chat_id,
                    message_id=os.urandom(16).hex(),
                    content="".join(self.tokens),
                    role="ai",
                    timestamp=datetime.utcnow(),
                )
                # Put message in queue
                asyncio.create_task(self.queue.put(streaming_message))

        # Initialize config with callbacks
        config = RunnableConfig(
            callbacks=[StreamingHandler(chat_id, message_queue)],
            configurable={
                "thread_id": chat_id,
                "user_id": user_id,
                "session_id": session_id,
                "language": language,
                "voice_type": voice_type,
            },
        )

        # Get chat history from Firestore
        chat_history = FirestoreChatMessageHistory(chat_id)
        history_messages = await chat_history.get_messages()

        # Create user message
        user_message = HumanMessage(
            content=message,
            additional_kwargs={
                "message_id": os.urandom(16).hex(),
                "user_id": user_id,
                "session_id": session_id,
                "chat_id": chat_id,
                "timestamp": datetime.utcnow(),
            },
        )

        # Add user message to Firestore history
        await chat_history.add_message(user_message)

        # Update chat metadata
        await update_chat_document(user_id, chat_id, session_id, message, redis_client)
        await manage_session(session_id, user_agent, client_ip, user_id)

        # Configure voice and language
        voice_prompts = {
            "normal": "You are an AI-powered chatbot specialized in assisting cannabis marketers. Your name is Smokey.",
            "pops": "You are a fatherly and upbeat AI assistant, ready to help with cannabis marketing. But you sound like Pops from the movie Friday, use his style of talk.",
            "smokey": "You are a laid-back and cool AI assistant, providing cannabis marketing insights. But sounds like Smokey from the movie Friday, use his style of talk.",
        }
        voice_prompt = voice_prompts.get(voice_type, voice_prompts["normal"])

        # Convert messages to format expected by agent
        formatted_messages = []
        for msg in history_messages:
            if isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_messages.append({"role": "assistant", "content": msg.content})

        # Create initial state
        current_state = MessagesState(
            messages=formatted_messages
            + [
                {"role": "system", "content": voice_prompt},
                {"role": "user", "content": message},
            ],
            metadata={
                "user_id": user_id,
                "session_id": session_id,
                "language": language,
                "voice_type": voice_type,
            },
            next_step="agent",
            agent_scratchpad=[],
            chat_id=chat_id,
        )

        # Get existing state or use current state
        try:
            existing_state = configurable_agent.get_state(config)
            if existing_state:
                # Merge the states
                existing_state["messages"].extend(
                    current_state["messages"][-2:]
                )  # Add new system and user messages
                current_state = existing_state
        except Exception:
            pass  # Use current_state if no existing state

        # Start agent processing in background
        agent_task = asyncio.create_task(
            configurable_agent.ainvoke(current_state, config=config)
        )

        # Stream messages while agent is processing
        while not agent_task.done():
            try:
                streaming_message = await asyncio.wait_for(
                    message_queue.get(), timeout=0.1
                )
                yield streaming_message
            except asyncio.TimeoutError:
                continue

        # Get final result
        result = await agent_task
        logger.debug(f"Agent task result: {result}")

        # Process response
        if not result.get("messages"):
            raise HTTPException(status_code=500, detail="No response from agent")

        ai_response = result["messages"][-1]["content"]
        logger.debug(f"Raw AI response: {ai_response}")
        response_text = ""
        data = None

        # Handle list response from product recommendation
        if isinstance(ai_response, str):
            try:
                # Try to parse the response as JSON in case it's a string representation of a list
                parsed_response = json.loads(ai_response)
                if isinstance(parsed_response, list) and len(parsed_response) == 2:
                    response_text = parsed_response[0]
                    data = parsed_response[1]
                else:
                    response_text = ai_response
            except json.JSONDecodeError:
                response_text = ai_response
        else:
            response_text = str(ai_response)

        logger.debug(f"Final response_text: {response_text}")
        logger.debug(f"Final data: {data}")

        # Create assistant message
        assistant_message = AIMessage(
            content=response_text,
            additional_kwargs={
                "message_id": os.urandom(16).hex(),
                "session_id": session_id,
                "chat_id": chat_id,
                "data": data,
                "timestamp": datetime.utcnow(),
            },
        )

        # Add assistant message to Firestore history
        await chat_history.add_message(assistant_message)

        # Schedule cleanup
        background_tasks.add_task(clean_up_old_sessions)

        # Convert to ChatMessage for response
        response_message = ChatMessage(
            message_id=assistant_message.additional_kwargs["message_id"],
            user_id=None,
            session_id=session_id,
            role="ai",
            content=response_text,
            chat_id=chat_id,
            data=data,
            timestamp=assistant_message.additional_kwargs["timestamp"],
        )

        yield response_message

    except Exception as e:
        logger.error(f"Error processing chat message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def update_chat_document(
    user_id: Optional[str],
    chat_id: str,
    session_id: str,
    message: str,
    redis_client: Redis,
):
    """Update or create chat document in Firestore."""
    chat_ref = db.collection("chats").document(chat_id)
    chat_doc = await chat_ref.get()

    if not chat_doc.exists:
        default_name = await generate_default_chat_name(message)
        chat_data = {
            "chat_id": chat_id,
            "user_ids": [user_id] if user_id else [],
            "session_ids": [session_id] if not user_id else [],
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
            "name": default_name,
        }
        await chat_ref.set(chat_data)
        if user_id:
            cache_key = f"user_chats:{user_id}:page:{1}:size:{20}"
            await redis_client.delete(cache_key)
        logger.debug(f"New chat document created with chat_id: {chat_id}")
    else:
        updates = {}
        if user_id and user_id not in chat_doc.to_dict().get("user_ids", []):
            updates["user_ids"] = firestore.ArrayUnion([user_id])
        if not user_id and session_id not in chat_doc.to_dict().get("session_ids", []):
            updates["session_ids"] = firestore.ArrayUnion([session_id])
        if updates:
            updates["updated_at"] = firestore.SERVER_TIMESTAMP
            await chat_ref.update(updates)
            logger.debug(f"Chat document updated with chat_id: {chat_id}")


async def manage_session(
    session_id: str,
    user_agent: str,
    client_ip: str,
    user_id: Optional[str],
):
    """Create or update session document in Firestore."""
    session_ref = db.collection("sessions").document(session_id)
    session_doc = await session_ref.get()
    if not session_doc.exists:
        session_data = {
            "session_id": session_id,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "user_agent": user_agent,
            "ip_address": client_ip,
            "user_id": user_id if user_id else None,
        }
        await session_ref.set(session_data, merge=True)
    elif user_id and user_id != session_doc.to_dict().get("user_id"):
        await session_ref.update({"user_id": user_id})


async def generate_default_chat_name(message: str) -> str:
    """Generate a default name for a new chat."""
    title_prompt = f"Generate a short, catchy title (max 5 words) for a chat that starts with this message: '{message}'"
    messages = [HumanMessage(content=title_prompt)]
    response = await llm.ainvoke(messages)
    return response.content.strip().strip("\"'")


@lru_cache(maxsize=100)
async def get_cached_chat_name(chat_id: str, redis_client: Redis) -> str:
    """Get chat name from cache or Firestore."""
    cached_name = await redis_client.get(f"chat_name:{chat_id}")
    if cached_name:
        return cached_name.decode("utf-8")

    chat_ref = db.collection("chats").document(chat_id)
    chat_doc = await chat_ref.get()
    name = (
        chat_doc.to_dict().get("name", "Unnamed Chat")
        if chat_doc.exists
        else "Unnamed Chat"
    )

    await redis_client.set(f"chat_name:{chat_id}", name, ex=3600)
    return name


async def rename_chat(chat_id: str, new_name: str, user_id: str) -> Dict[str, str]:
    """Rename a chat if the user has permission."""
    try:
        if not new_name:
            raise HTTPException(status_code=400, detail="New name must be provided")

        chat_ref = db.collection("chats").document(chat_id)
        chat_doc = await chat_ref.get()

        if not chat_doc.exists:
            raise HTTPException(status_code=404, detail="Chat not found")

        chat_data = chat_doc.to_dict()
        if user_id not in chat_data.get("user_ids", []):
            raise HTTPException(
                status_code=403, detail="User not authorized to rename this chat"
            )

        await chat_ref.update(
            {"name": new_name, "updated_at": firestore.SERVER_TIMESTAMP}
        )
        logger.debug(f"Chat {chat_id} renamed to {new_name} by user {user_id}")

        return {"message": "Chat renamed successfully"}

    except HTTPException as http_ex:
        logger.error(f"HTTP error occurred: {http_ex.detail}")
        raise
    except Exception as e:
        logger.error(f"Error occurred while renaming chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def archive_chat(chat_id: str, user_id: str) -> Dict[str, str]:
    """Archive a chat if the user has permission."""
    try:
        chat_ref = db.collection("chats").document(chat_id)
        chat_doc = await chat_ref.get()

        if not chat_doc.exists:
            raise HTTPException(status_code=404, detail="Chat not found")

        chat_data = chat_doc.to_dict()
        if user_id not in chat_data.get("user_ids", []):
            raise HTTPException(
                status_code=403, detail="User not authorized to archive this chat"
            )

        await chat_ref.update(
            {"archived": True, "updated_at": firestore.SERVER_TIMESTAMP}
        )
        logger.debug(f"Chat {chat_id} archived by user {user_id}")

        return {"message": "Chat archived successfully"}

    except Exception as e:
        logger.error(f"Error occurred while archiving chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def delete_chat(chat_id: str, user_id: str) -> Dict[str, str]:
    """Delete a chat and its messages if the user has permission."""
    try:
        chat_ref = db.collection("chats").document(chat_id)
        chat_doc = await chat_ref.get()

        if not chat_doc.exists:
            raise HTTPException(status_code=404, detail="Chat not found")

        chat_data = chat_doc.to_dict()
        if user_id not in chat_data.get("user_ids", []):
            raise HTTPException(
                status_code=403, detail="User not authorized to delete this chat"
            )

        messages_ref = chat_ref.collection("messages")
        batch = db.batch()
        messages = await messages_ref.get()

        for msg in messages:
            batch.delete(msg.reference)
        batch.delete(chat_ref)
        await batch.commit()

        logger.debug(f"Chat {chat_id} and its messages deleted by user {user_id}")
        return {"message": "Chat deleted successfully"}

    except Exception as e:
        logger.error(f"Error occurred while deleting chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_chat_messages(
    chat_id: str, limit: Optional[int] = None
) -> Dict[str, Any]:
    """Get all messages for a chat."""
    logger.debug(f"Fetching chat messages for chat_id: {chat_id}")
    try:
        chat_history = FirestoreChatMessageHistory(chat_id)
        messages = await chat_history.get_messages()

        # Convert LangChain messages to ChatMessage
        chat_messages = []
        for msg in messages:
            chat_message = ChatMessage(
                message_id=msg.additional_kwargs.get("message_id"),
                user_id=msg.additional_kwargs.get("user_id"),
                session_id=msg.additional_kwargs.get("session_id"),
                role="human" if isinstance(msg, HumanMessage) else "ai",
                content=msg.content,
                chat_id=msg.additional_kwargs.get("chat_id"),
                data=msg.additional_kwargs.get("data"),
                timestamp=msg.additional_kwargs.get("timestamp"),
            )
            chat_messages.append(chat_message)

        return {"chat_id": chat_id, "messages": chat_messages}
    except Exception as e:
        logger.error(f"Error getting chat messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def record_feedback(
    user_id: str, message_id: str, feedback_type: str
) -> Dict[str, str]:
    """Record user feedback for a message."""
    feedback_ref = db.collection("feedback").document()
    feedback_data = {
        "user_id": user_id,
        "message_id": message_id,
        "feedback_type": feedback_type,
        "timestamp": firestore.SERVER_TIMESTAMP,
    }
    await feedback_ref.set(feedback_data)
    return {"status": "success", "feedback_id": feedback_ref.id}


async def retry_message(
    user_id: str,
    message_id: str,
    background_tasks: BackgroundTasks,
    redis: Redis,
) -> AsyncGenerator[ChatMessage, None]:
    """Retry processing a specific message."""
    try:
        # Find the message in all chats
        chats_ref = db.collection("chats")
        chats = await chats_ref.where("user_ids", "array_contains", user_id).get()

        for chat in chats:
            messages_ref = chat.reference.collection("messages")
            message_doc = await messages_ref.where(
                "additional_kwargs.message_id", "==", message_id
            ).get()

            if message_doc:
                # Get the original message
                original_message = message_doc[0].to_dict()

                # Get the previous human message
                previous_messages = (
                    await messages_ref.where(
                        "timestamp", "<", original_message["timestamp"]
                    )
                    .where("type", "==", "human")
                    .order_by("timestamp", direction=firestore.Query.DESCENDING)
                    .limit(1)
                    .get()
                )

                if previous_messages:
                    prev_message = previous_messages[0].to_dict()

                    # Get the full chat history up to this point
                    chat_history = FirestoreChatMessageHistory(chat.id)
                    history_messages = await chat_history.get_messages()

                    # Filter messages up to the retry point
                    filtered_messages = []
                    for msg in history_messages:
                        if (
                            msg.additional_kwargs.get("timestamp")
                            < original_message["timestamp"]
                        ):
                            if isinstance(msg, SystemMessage):
                                filtered_messages.append(
                                    {"role": "system", "content": msg.content}
                                )
                            elif isinstance(msg, HumanMessage):
                                filtered_messages.append(
                                    {"role": "user", "content": msg.content}
                                )
                            elif isinstance(msg, AIMessage):
                                filtered_messages.append(
                                    {"role": "assistant", "content": msg.content}
                                )

                    # Create initial state for retry
                    initial_state = MessagesState(
                        messages=filtered_messages,
                        metadata={
                            "user_id": user_id,
                            "session_id": prev_message["additional_kwargs"].get(
                                "session_id", ""
                            ),
                            "language": "English",
                            "voice_type": "normal",
                        },
                        next_step="agent",
                        agent_scratchpad=[],
                        chat_id=chat.id,
                    )

                    async for response in process_chat_message(
                        user_id=user_id,
                        chat_id=chat.id,
                        session_id=prev_message["additional_kwargs"].get(
                            "session_id", ""
                        ),
                        client_ip="",
                        message=prev_message["content"],
                        user_agent="",
                        voice_type="normal",
                        background_tasks=background_tasks,
                        redis_client=redis,
                    ):
                        yield response
                    return

        raise HTTPException(
            status_code=404, detail="Message not found or not eligible for retry"
        )

    except Exception as e:
        logger.error(f"Error retrying message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
