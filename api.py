from fastapi import FastAPI, HTTPException, Request, Header
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
import json
from pydantic import BaseModel
from typing import Dict, List
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from dotenv import load_dotenv
import os
import logging
import firebase_admin
from firebase_admin import credentials, firestore, auth
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()  # Load environment variables from .env


def initialize_firebase():
    """Initialize Firebase using credentials from the environment."""
    if not firebase_admin._apps:
        cred_input = os.getenv("FIREBASE_CREDENTIALS")
        if not cred_input:
            raise ValueError("FIREBASE_CREDENTIALS environment variable is not set")

        try:
            cred_json = json.loads(cred_input)
            cred = credentials.Certificate(cred_json)
        except json.JSONDecodeError:
            if not os.path.exists(cred_input):
                raise FileNotFoundError(f"Credential path {cred_input} does not exist")
            cred = credentials.Certificate(cred_input)

        firebase_admin.initialize_app(cred)


initialize_firebase()
db = firestore.client()

# Simple in-memory cache
session_cache = {}


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
app.add_middleware(
    SessionMiddleware, secret_key=os.getenv("SESSION_SECRET_KEY", "your-secret-key")
)
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
llm = OpenAI(model="gpt-4o-mini")
ft_model = "ft:gpt-3.5-turbo-0125:bakedbot::9rOlft9b"
ft_llm = OpenAI(model=ft_model)


# Function to get query engine
def get_query_engine(path):
    vector_store = FaissVectorStore.from_persist_dir(path)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=path
    )
    index = load_index_from_storage(
        storage_context=storage_context, embed_model=embed_model
    )
    query_engine = index.as_query_engine()
    return query_engine


# Initialize query engines
compliance_guidelines = get_query_engine("data/Compliance guidelines")
marketing_strategies = get_query_engine("data/Marketing strategies and best practices")
seasonal_marketing = get_query_engine("data/Seasonal and holiday marketing plans")
state_policies = get_query_engine("data/State-specific cannabis marketing regulations")

# Define the query engine tools with detailed descriptions
compliance_guidelines_tool = QueryEngineTool(
    query_engine=compliance_guidelines,
    metadata=ToolMetadata(
        name="Compliance_Guidelines",
        description="Provides guidelines on compliance requirements for cannabis marketing across various regions.",
    ),
)

marketing_strategies_tool = QueryEngineTool(
    query_engine=marketing_strategies,
    metadata=ToolMetadata(
        name="Marketing_Strategies",
        description="Offers strategies and best practices for effective cannabis marketing.",
    ),
)

seasonal_marketing_tool = QueryEngineTool(
    query_engine=seasonal_marketing,
    metadata=ToolMetadata(
        name="Seasonal_Marketing",
        description="Provides marketing plans and strategies tailored for seasonal and holiday events. ",
    ),
)

state_policies_tool = QueryEngineTool(
    query_engine=state_policies,
    metadata=ToolMetadata(
        name="State_Policies",
        description="Details state-specific cannabis marketing regulations and policies.",
    ),
)


# Define custom tools
def generate_campaign_planner(template_name: str) -> str:
    return f"<h2>Campaign Planner Template: {template_name}</h2><p>[Template content specific to {template_name}]</p>"


def calculate_roi(investment: float, revenue: float) -> float:
    return (revenue - investment) / investment


def generate_compliance_checklist(state: str) -> str:
    return f"<h2>Compliance Checklist for {state}</h2><ul><li>Checklist item 1</li><li>Checklist item 2</li><li>Checklist item 3</li></ul>"


# Define the custom tools
campaign_planner_tool = FunctionTool.from_defaults(
    fn=generate_campaign_planner,
    name="GenerateCampaignPlanner",
    description="Generates a campaign planning template based on the user's requirements.",
)

roi_calculator_tool = FunctionTool.from_defaults(
    fn=calculate_roi,
    name="CalculateROI",
    description="Calculates the return on investment (ROI) for a given investment and revenue.",
)

compliance_checklist_tool = FunctionTool.from_defaults(
    fn=generate_compliance_checklist,
    name="GenerateComplianceChecklist",
    description="Generates a compliance checklist for the specified state.",
)


# Define the fine-tuned LLM function
def recommend_cannabis_strain(question: str) -> str:
    """
    Recommend a cannabis strain based on the given question.

    Attributes:
        question (str): A detailed question containing attributes such as type, rating, effects, and flavor.

    Returns:
        str: A detailed recommendation of a cannabis strain.
    """
    messages = [
        ChatMessage(
            role="system",
            content="You are an expert in cannabis marketing, providing detailed and personalized recommendations.",
        ),
        ChatMessage(role="user", content=question),
    ]
    resp = llm.chat(messages)
    answer = resp.message.content
    return answer


# Create the FunctionTool with a detailed description
recommend_cannabis_strain_tool = FunctionTool.from_defaults(
    recommend_cannabis_strain,
    name="CannabisStrainRecommendation",
    description=(
        "This tool recommends a cannabis strain with details based on a detailed user question. "
    ),
)

# Combine all tools
tools = [
    compliance_guidelines_tool,
    marketing_strategies_tool,
    recommend_cannabis_strain_tool,
    seasonal_marketing_tool,
    state_policies_tool,
    # roi_calculator_tool,
    campaign_planner_tool,
    compliance_checklist_tool,
]

system_prompt = """
You are an AI-powered chatbot specialized in assisting cannabis marketers with strategy, compliance, and campaign planning.
Your main objectives are to provide accurate, up-to-date information on cannabis marketing regulations across different US states and Canada,
offer strategic marketing advice tailored to the user's experience level, assist in campaign planning including seasonal and holiday-specific strategies,
and ensure all advice adheres to legal and ethical standards in cannabis marketing.

Additionally, you have the ability to recommend cannabis strains based on specific user-provided attributes such as type, rating, desired effects, and preferred flavors. 
The personalized strain recommendation tool helps users find the best cannabis strains that match their preferences and needs.

"""

agent = OpenAIAgent.from_tools(
    tools=tools,
    llm=llm,
    verbose=True,
    system_prompt=system_prompt,
)


def verify_firebase_token(token: str):
    """Verify Firebase authentication token."""
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication token")


class ChatRequest(BaseModel):
    message: str
    voice_type: str = "normal"
    chat_id: str = None  # Optional chat ID for authenticated users


class ChatResponse(BaseModel):
    response: str


# Helper functions for managing conversation context
def get_conversation_context(session_ref, max_context_length=5):
    session_doc = session_ref.get()
    if session_doc.exists:
        context = session_doc.to_dict().get("context", [])
        return context[-max_context_length:]
    return []


def update_conversation_context(session_ref, context):
    session_ref.set({"context": context})


def summarize_context(context):
    summary_prompt = "Summarize the following conversation context briefly: "
    full_context = " ".join([f"{msg['role']}: {msg['content']}" for msg in context])
    summarization_input = summary_prompt + full_context

    summary = llm.chat([ChatMessage(role="system", content=summarization_input)])
    return [{"role": "system", "content": summary.message.content}]


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    fastapi_request: Request,
    authorization: str = Header(None),  # Firebase auth token
):
    try:
        logger.debug("Received chat request: %s", request)
        user_id = None
        chat_id = request.chat_id
        session_ref = None

        if authorization:
            try:
                token = authorization.split(" ")[1]
                decoded_token = verify_firebase_token(token)
                user_id = decoded_token.get("uid")
            except IndexError:
                raise HTTPException(
                    status_code=400, detail="Invalid authorization header format"
                )

        if user_id and chat_id:
            session_ref = db.collection("user_chats").document(chat_id)
            logger.debug(
                "User history found for user_id: %s, chat_id: %s", user_id, chat_id
            )
        else:
            session_id = fastapi_request.session.get("session_id")
            if not session_id:
                session_id = os.urandom(16).hex()
                fastapi_request.session["session_id"] = session_id
            logger.debug("Session history found for session_id: %s", session_id)
            session_ref = db.collection("sessions").document(session_id)

        context = get_conversation_context(session_ref)
        logger.debug("Initial context: %s", context)
        context.append({"role": "user", "content": request.message})

        if len(context) > 10:  # Adjust this threshold as necessary
            context = summarize_context(context)
        else:
            context = context[-10:]  # Ensure context does not grow indefinitely

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

        # The message to be sent to the agent
        new_prompt = f"{voice_prompt} Instructions: {request.message}. Always OUTPUT in markdown."

        logger.debug("New prompt for agent: %s", new_prompt)
        agent_response = agent.chat(
            message=new_prompt, chat_history=chat_history  # Provide the chat history
        )

        logger.debug("Agent response: %s", agent_response)
        response_text = (
            agent_response.response if agent_response else "No response available."
        )

        # Update the context with the assistant's response
        context.append({"role": "assistant", "content": response_text})

        update_conversation_context(session_ref, context)

        return ChatResponse(response=response_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the FastAPI app with Uvicorn for performance
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
