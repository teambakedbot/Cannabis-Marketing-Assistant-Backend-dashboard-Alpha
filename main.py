from fastapi import FastAPI, HTTPException, Request, Header, BackgroundTasks, Query
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from firebase_utils import db, verify_firebase_token, firestore, auth
from models import ChatRequest, ChatResponse
from tools import agent, ChatMessage
from utils import (
    get_conversation_context,
    update_conversation_context,
    summarize_context,
    clean_up_old_sessions,
    merge_unauthenticated_session_to_user,
    end_session,
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    SessionMiddleware, secret_key=os.getenv("SESSION_SECRET_KEY", "your-secret-key")
)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    fastapi_request: Request,
    background_tasks: BackgroundTasks,
    authorization: str = Header(None),  # Firebase auth token
):
    logger.debug("Starting chat_endpoint")
    try:
        user_id = None
        chat_id = request.chat_id

        # Retrieve or create session_id
        session_id = fastapi_request.session.get("session_id")
        if not session_id:
            session_id = os.urandom(16).hex()
            fastapi_request.session["session_id"] = session_id
            logger.debug(f"New session created with session_id: {session_id}")
        else:
            logger.debug(f"Existing session retrieved with session_id: {session_id}")

        client_ip = fastapi_request.client.host
        user_agent = fastapi_request.headers.get("user-agent", "unknown")

        # Verify user authentication and get user_id if authenticated
        if authorization:
            try:
                token = authorization.split(" ")[1]
                decoded_token = verify_firebase_token(token)
                user_id = decoded_token.get("uid")

                # If chat_id is not provided, generate a new one
                if not chat_id:
                    chat_id = os.urandom(16).hex()
                    logger.debug(f"Generated new chat_id: {chat_id}")
            except IndexError:
                raise HTTPException(
                    status_code=400, detail="Invalid authorization header format"
                )
        else:
            # For anonymous users, generate a new chat_id if none is provided
            if not chat_id:
                chat_id = os.urandom(16).hex()
                logger.debug(f"Generated new chat_id for anonymous session: {chat_id}")

        # Reference the chat document in the chats collection
        chat_ref = db.collection("chats").document(chat_id)

        # If the chat is new, set up the chat document
        chat_doc = chat_ref.get()
        if not chat_doc.exists:
            chat_data = {
                "chat_id": chat_id,
                "user_ids": [user_id] if user_id else [],
                "session_ids": [session_id] if not user_id else [],
                "created_at": firestore.SERVER_TIMESTAMP,
                "last_updated": firestore.SERVER_TIMESTAMP,
            }
            chat_ref.set(chat_data)
            logger.debug(f"New chat document created with chat_id: {chat_id}")
        else:
            chat_data = chat_doc.to_dict()
            if user_id and user_id not in chat_data["user_ids"]:
                chat_ref.update({"user_ids": firestore.ArrayUnion([user_id])})
            if not user_id and session_id not in chat_data["session_ids"]:
                chat_ref.update({"session_ids": firestore.ArrayUnion([session_id])})
            logger.debug(f"Chat document updated with chat_id: {chat_id}")

        session_ref = db.collection("sessions").document(session_id)
        session_data = {
            "session_id": session_id,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "user_agent": user_agent,
            "ip_address": client_ip,
            "user_id": user_id if user_id else None,
        }

        # Create or update the session document
        session_ref.set(session_data, merge=True)

        # Retrieve the conversation context (up to the last 10 messages)
        messages_ref = chat_ref.collection("messages")
        query = messages_ref.order_by(
            "timestamp", direction=firestore.Query.DESCENDING
        ).limit(10)
        messages = query.stream()
        context = [msg.to_dict() for msg in messages][::-1]

        new_message = {
            "message_id": os.urandom(16).hex(),
            "user_id": user_id if user_id else None,
            "session_id": session_id,
            "role": "user",
            "content": request.message,
            "timestamp": firestore.SERVER_TIMESTAMP,
        }
        context.append(new_message)

        # Optionally summarize or trim the context if it exceeds a certain length
        if len(context) > 10:
            context = summarize_context(context)
        else:
            context = context[-10:]

        # Convert context to ChatMessage objects
        chat_history = [
            ChatMessage(role=msg["role"], content=msg["content"]) for msg in context
        ]

        # Define voice types
        voice_prompts = {
            "normal": "You are an AI-powered chatbot specialized in assisting cannabis marketers. Your name is BakedBot.",
            "pops": "You are a fatherly and upbeat AI assistant, ready to help with cannabis marketing. But you sound like Pops from the movie Friday, use his style of talk.",
            "smokey": "You are a laid-back and cool AI assistant, providing cannabis marketing insights. But sounds like Smokey from the movie Friday, use his style of talk.",
        }
        voice_prompt = voice_prompts.get(request.voice_type, voice_prompts["normal"])

        # Create the prompt to send to the AI agent
        new_prompt = f"{voice_prompt} Instructions: {request.message}. Always OUTPUT in markdown."
        agent_response = agent.chat(message=new_prompt, chat_history=chat_history)

        response_text = (
            agent_response.response if agent_response else "No response available."
        )

        # Save the assistant's response in the chat history
        assistant_message = {
            "message_id": os.urandom(16).hex(),
            "user_id": None,
            "session_id": session_id,
            "role": "assistant",
            "content": response_text,
            "timestamp": firestore.SERVER_TIMESTAMP,
        }
        messages_ref.document(assistant_message["message_id"]).set(assistant_message)

        # Also store the new user message in the messages collection
        messages_ref.document(new_message["message_id"]).set(new_message)

        # Update the last_updated timestamp on the chat document
        chat_ref.update({"last_updated": firestore.SERVER_TIMESTAMP})

        # Schedule the cleanup task for old sessions
        background_tasks.add_task(clean_up_old_sessions)

        # Return the chat response along with the chat_id
        return ChatResponse(response=response_text, chat_id=chat_id)

    except Exception as e:
        logger.error("Error occurred in chat_endpoint: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user/chats")
async def get_user_chats(
    authorization: str = Header(None),  # Firebase auth token
):
    logger.debug("Fetching all user chats")
    try:
        # Verify user authentication
        if authorization:
            try:
                token = authorization.split(" ")[1]
                decoded_token = verify_firebase_token(token)
                user_id = decoded_token.get("uid")
                logger.debug("User authenticated with user_id: %s", user_id)
            except IndexError:
                raise HTTPException(
                    status_code=400, detail="Invalid authorization header format"
                )
        else:
            raise HTTPException(
                status_code=401, detail="Authorization header missing or invalid"
            )

        # Fetch all chats for the authenticated user
        chats_ref = db.collection("chats").where("user_ids", "array_contains", user_id)
        chats = chats_ref.stream()
        chat_list = [{"chat_id": chat.id, **chat.to_dict()} for chat in chats]

        logger.debug("User chats retrieved for user_id: %s", user_id)
        return {"user_id": user_id, "chats": chat_list}

    except Exception as e:
        logger.error("Error occurred while fetching user chats: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/messages")
async def get_chat_messages(
    chat_id: str = Query(..., description="The chat ID to fetch messages for"),
    authorization: str = Header(None),  # Firebase auth token
):
    logger.debug("Fetching chat messages for chat_id: %s", chat_id)
    try:
        # Verify user authentication
        if authorization:
            try:
                token = authorization.split(" ")[1]
                decoded_token = verify_firebase_token(token)
                user_id = decoded_token.get("uid")
                logger.debug("User authenticated with user_id: %s", user_id)
            except IndexError:
                raise HTTPException(
                    status_code=400, detail="Invalid authorization header format"
                )
        else:
            raise HTTPException(
                status_code=401, detail="Authorization header missing or invalid"
            )

        # Reference the chat document in the chats collection
        chat_ref = db.collection("chats").document(chat_id)
        chat_doc = chat_ref.get()
        if not chat_doc.exists:
            raise HTTPException(status_code=404, detail="Chat not found")

        # Retrieve messages from the chat
        messages_ref = chat_ref.collection("messages")
        messages = messages_ref.order_by("timestamp").stream()
        chat_history = [msg.to_dict() for msg in messages]

        logger.debug("Chat messages retrieved for chat_id: %s", chat_id)
        return {"chat_id": chat_id, "messages": chat_history}

    except Exception as e:
        logger.error("Error occurred while fetching chat messages: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/logout")
async def logout_endpoint(
    fastapi_request: Request,
    background_tasks: BackgroundTasks,
    authorization: str = Header(None),
):
    try:
        session_id = fastapi_request.session.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID not found")

        if authorization:
            try:
                token = authorization.split(" ")[1]
                decoded_token = verify_firebase_token(token)
                user_id = decoded_token.get("uid")

                # Revoke Firebase token
                auth.revoke_refresh_tokens(user_id)
                logger.debug(f"Revoked Firebase token for user_id: {user_id}")

                # End session and calculate session length
                end_session(session_id)

                # Optionally clear the session from the request
                fastapi_request.session.clear()
                logger.debug(f"Session {session_id} ended and cleared.")

                # Schedule the cleanup task for old sessions
                background_tasks.add_task(clean_up_old_sessions)

                return {"detail": "Logged out and session ended successfully."}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        else:
            raise HTTPException(
                status_code=400, detail="Authorization header missing or invalid"
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the FastAPI app with Uvicorn for performance
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
